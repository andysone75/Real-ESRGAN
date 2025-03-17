import argparse
import cv2
import glob
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from colorama import Fore, Style, init

# Инициализация colorama для цветного вывода
init(autoreset=True)


def process_image(path, args, upsampler, face_enhancer=None, progress_bar=None):
    """Обрабатывает одно изображение и сохраняет результат."""
    imgname, extension = os.path.splitext(os.path.basename(path))
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None

    try:
        if args.face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=args.outscale)
    except RuntimeError as error:
        if progress_bar:
            progress_bar.update(1)
        return None, imgname

    if args.ext == 'auto':
        extension = extension[1:]
    else:
        extension = args.ext
    if img_mode == 'RGBA':  # RGBA images should be saved in png format
        extension = 'png'
    if args.suffix == '':
        save_path = os.path.join(args.output, f'{imgname}.{extension}')
    else:
        save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
    cv2.imwrite(save_path, output)
    if progress_bar:
        progress_bar.update(1)
    return save_path, imgname


def process_on_gpu(paths, args, gpu_id, progress_bar):
    """Обрабатывает список изображений на указанном GPU."""
    # Устанавливаем устройство (GPU)
    torch.cuda.set_device(gpu_id)

    # Инициализация модели для текущего GPU
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4

    # Загрузка модели
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]
            for url in file_url:
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # Использование DNI для управления уровнем шума
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # Инициализация RealESRGANer для текущего GPU
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=gpu_id)

    # Инициализация GFPGAN (если нужно)
    face_enhancer = None
    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    # Обработка изображений на текущем GPU
    for path in paths:
        save_path, imgname = process_image(path, args, upsampler, face_enhancer, progress_bar)
        if save_path:
            print(Fore.GREEN + f"GPU {gpu_id}: Saved {save_path}")
        else:
            print(Fore.RED + f"GPU {gpu_id}: Failed {imgname}")


def main():
    """Inference demo for Real-ESRGAN with multi-GPU support and progress bars."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
    parser.add_argument(
        '--num_threads', type=int, default=4, help='Number of threads for parallel processing')

    args = parser.parse_args()

    # Создание выходной папки
    os.makedirs(args.output, exist_ok=True)

    # Получение списка изображений
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    # Определение количества доступных GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices found!")

    # Разделение задач между GPU
    paths_per_gpu = [paths[i::num_gpus] for i in range(num_gpus)]

    # Создание прогресс-баров для каждого GPU
    progress_bars = [tqdm(total=len(paths_per_gpu[i]), desc=f"GPU {i}", unit="image", position=i) for i in range(num_gpus)]

    # Запуск обработки на каждом GPU
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id, paths in enumerate(paths_per_gpu):
            futures.append(executor.submit(process_on_gpu, paths, args, gpu_id, progress_bars[gpu_id]))

        for future in as_completed(futures):
            future.result()  # Ожидание завершения всех задач

    # Закрытие всех прогресс-баров
    for pbar in progress_bars:
        pbar.close()

if __name__ == '__main__':
    main()