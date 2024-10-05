import copy
import time
import torch
import numpy as np

from typing import Optional

from tqdm import tqdm

from pathlib import Path
from pyldpc import make_ldpc, encode, decode, get_message
from utils.utils_bit import bits_from_file, bits_to_file


class Watermark:
    BIT_TO_SIGNAL_MAPPING = {
        1: -1,
        0: 1
    }

    def __init__(self, seed: int, ldpc_seed: int, parameter_seed: int, ratio: float, device: str,
                 error_correction: bool, watermark_path: Path,
                 result_path: Path, logger):
        self.seed = seed
        self.parameter_seed = parameter_seed
        self.ratio = ratio
        self.device = device
        self.error_correction = error_correction
        self.H = None
        self.G = None
        # self.preamble = None
        self.watermark_path = watermark_path
        self.result_path = result_path
        self.message = bits_from_file(watermark_path)
        self.watermark_length = 0
        self.logger = logger
        self.ldpc_seed = ldpc_seed

        np.random.seed(self.ldpc_seed)
        self.preamble = np.sign(np.random.uniform(-1, 1, 200))

        if self.error_correction:
            if len(self.message) > 4000:
                k = 3048
            else:
                k = 96
            d_v = 3
            d_c = 12
            n = k * int(d_c / d_v)
            self.H, self.G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True, seed=ldpc_seed)

    def get_message_length(self):
        k = self.G.shape[1]

        snr1 = 10000000000000000
        c = []
        remaining_bits = len(self.message) % k
        n_chunks = int(len(self.message) / k)
        chunks = list()

        for ch in range(n_chunks):
            chunks.append(self.message[ch * k:ch * k + k])

        encoded = map(lambda x: encode(self.G, x, snr1), chunks)
        for enc in encoded:
            c.extend(enc)

        last_part = []
        last_part.extend(self.message[n_chunks * k:])
        last_part.extend([0] * (k - remaining_bits))

        c.extend(encode(self.G, last_part, snr1))

        # np.random.seed(self.ldpc_seed)
        preamble = np.sign(np.random.uniform(-1, 1, 200))
        b = np.concatenate((preamble, c))

        return len(b)

    def embed(self, model, gamma: Optional[float] = None):
        start = time.time()

        model_st_dict = model.state_dict()
        models_w = []
        layer_lengths = dict()

        layers = [n for n in model_st_dict.keys() if "weight" in str(n)][:-1]
        for layer in layers:
            x = model_st_dict[layer].detach().cpu().numpy().flatten()
            layer_lengths[layer] = len(x)
            models_w.extend(list(x))

        spreading_code_length = int(len(models_w) * self.ratio)

        np.random.seed(self.parameter_seed)
        filter_indexes = np.random.randint(0, len(models_w), spreading_code_length, np.int32).tolist()

        gradients = np.array(models_w)
        models_w = np.array(models_w)

        np.random.seed(self.ldpc_seed)
        if self.error_correction:
            k = self.G.shape[1]

            snr1 = 10000000000000000
            c = []
            remaining_bits = len(self.message) % k
            chunks = int(len(self.message) / k)

            for ch in range(chunks):
                c.extend(encode(self.G, self.message[ch * k:ch * k + k], snr1))

            last_part = []
            last_part.extend(self.message[chunks * k:])
            last_part.extend([0] * (k - remaining_bits))

            c.extend(encode(self.G, last_part, snr1))
            b = np.concatenate((self.preamble, c))
        else:
            b = [self.BIT_TO_SIGNAL_MAPPING[int(bit)] for bit in self.message]

        self.watermark_length = len(b)

        self.logger.info(f'Tattooing on {spreading_code_length} parameters')
        with tqdm(total=len(b)) as bar:
            bar.set_description('Watermarking')
            np.random.seed(self.seed)
            for bit in b:
                spreading_code = np.random.choice([-1, 1], size=spreading_code_length)
                current_bit_cdma_signal = gamma * spreading_code * bit
                models_w[filter_indexes] = np.add(models_w[filter_indexes], current_bit_cdma_signal)
                bar.update(1)

        curr_index = 0
        for layer in layers:
            x = np.array(models_w[curr_index:curr_index + layer_lengths[layer]])
            model_st_dict[layer] = torch.from_numpy(np.reshape(x, model_st_dict[layer].shape)).to(self.device)
            curr_index = curr_index + layer_lengths[layer]

        end = time.time()
        self.logger.info(f'Time to mark {end-start}')
        return model_st_dict

    def extract(self, model):
        extraction_path = self.result_path / 'watermarks'
        extraction_path.mkdir(parents=True, exist_ok=True)

        start = time.time()
        st_dict = model.state_dict()

        models_w_curr = []

        layer_lengths = dict()
        total_params = 0

        layers = [n for n in st_dict.keys() if "weight" in str(n)][:-1]
        for layer in layers:
            x_curr = st_dict[layer].detach().cpu().numpy().flatten()
            models_w_curr.extend(list(x_curr))
            layer_lengths[layer] = len(x_curr)
            total_params += len(x_curr)

        spreading_code_length = int(len(models_w_curr) * self.ratio)

        np.random.seed(self.parameter_seed)
        filter_indexes = np.random.randint(0, len(models_w_curr), spreading_code_length, np.int32)

        models_w_curr = np.array(models_w_curr)
        models_w_delta = models_w_curr[filter_indexes]
        spreading_code_length = len(models_w_delta)

        x = []
        ys = []

        if self.watermark_length == 0:
            self.watermark_length = self.get_message_length()
            print('The watermark length: ', self.watermark_length)
        np.random.seed(self.seed)
        for _ in range(self.watermark_length):
            spreading_code = np.random.choice([-1, 1], size=spreading_code_length)
            y_i = np.matmul(spreading_code.T, models_w_delta)
            ys.append(y_i)
            if not self.error_correction:
                x.append(0 if y_i > 0 else 1)

        # np.random.seed(self.ldpc_seed)
        if self.error_correction:
            y = np.array(ys)
            gain = np.mean(np.multiply(y[:200], self.preamble))
            sigma = np.std(np.multiply(y[:200], self.preamble) / gain)
            snr = -20 * np.log10(sigma)
            self.logger.info(f'Signal to Noise Ratio = {snr}')

            k = self.G.shape[0]
            y = y[200:]
            chunks = int(len(y) / k)

            for ch in range(chunks):
                d = decode(self.H, y[ch * k:ch * k + k] / gain, snr)
                x.extend(get_message(self.G, d))

        end = time.time()
        self.logger.info(f'Time to verify {end-start}')

        bits_to_file(extraction_path / self.watermark_path.parts[-1], x[:len(self.message)])

        str_wtm = ''.join(str(l) for l in x[:len(self.message)])
        str_msg = ''.join(str(l) for l in self.message)
        BER = float(len([i for i in range(len(str_wtm)) if str_wtm[i] != str_msg[i]])) / float(len(self.message))
        self.logger.info(f'Error Rate = {BER}')

        return True if BER == 0 else False
