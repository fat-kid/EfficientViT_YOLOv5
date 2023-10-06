# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
utils/initialization
"""

import contextlib
import platform
import threading


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


class TryExcept(contextlib.ContextDecorator):
    # YOLOv5 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=''):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def join_threads(verbose=False):
    # Join all daemon threads, i.e. atexit.register(lambda: join_threads())
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f'Joining thread {t.name}')
            t.join()


def notebook_init(verbose=True):
    # Check system software and hardware
    print('Checking setup...')

    import os
    import shutil

    from utils.general import check_font, check_requirements, is_colab
    from utils.torch_utils import select_device  # imports

    check_font()

    import psutil

    if is_colab():
        shutil.rmtree('/content/sample_data', ignore_errors=True)  # remove colab /sample_data directory

    # System info
    display = None
    if verbose:
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage('/')
        with contextlib.suppress(Exception):  # clear display if ipython is installed
            from IPython import display
            display.clear_output()
        s = f'({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)'
    else:
        s = ''

    select_device(newline=False)
    print(emojis(f'Setup complete ✅ {s}'))
    return display


# axs[0, 0].plot(x,box_loss_1,label='yolov5s')
# axs[0, 0].plot(x,box_loss_2,label='yolov5s_e_ciou')
# axs[0, 0].plot(x,box_loss_3,label='yolov5s_e_eiou')
# axs[0, 0].plot(x,box_loss_4,label='yolov5s_e_aeiou')
# axs[0, 0].set_title('box_loss')
# axs[0, 0].legend()
#
# axs[0, 1].plot(x,obj_loss_1,label='yolov5s_e_aeiou')
# axs[0, 1].plot(x,obj_loss_2,label='yolov5s')
# axs[0, 1].plot(x,obj_loss_3,label='yolov5s_e_eiou')
#
# axs[0, 1].set_title('obj_loss')
# axs[0, 1].legend()
#
# axs[0, 2].plot(x,val_box_loss_1,label='yolov5s_e_aeiou')
# axs[0, 2].plot(x,val_box_loss_2,label='yolov5s')
# axs[0, 2].plot(x,val_box_loss_3,label='yolov5s_e_eiou')
# axs[0, 2].set_title('val_box_loss')
#
# axs[0, 2].legend()
#
# axs[0, 3].plot(x,val_obj_loss_1,label='yolov5s_e_aeiou')
# axs[0, 3].plot(x,val_obj_loss_2,label='yolov5s')
# axs[0, 3].plot(x,val_obj_loss_3,label='yolov5s_e_eiou')
# axs[0, 3].set_title('val_obj_loss')
# axs[0, 3].legend()
#
# axs[1, 0].plot(x,precision_1,label='yolov5s_e_aeiou')
# axs[1, 0].plot(x,precision_2,label='yolov5s')
# axs[1, 0].plot(x,precision_3,label='yolov5s_e_eiou')
# axs[1, 0].set_title('precision')
# axs[1, 0].legend()
#
# axs[1, 1].plot(x,recall_1,label='yolov5s_e_aeiou')
# axs[1, 1].plot(x,recall_2,label='yolov5s')
# axs[1, 1].plot(x,recall_3,label='yolov5s_e_eiou')
# axs[1, 1].set_title('recall')
# axs[1, 1].legend()
#
# axs[1, 2].plot(x,mAP_05_1,label='yolov5s_e_aeiou')
# axs[1, 2].plot(x,mAP_05_2,label='yolov5s')
# axs[1, 2].plot(x,mAP_05_3,label='yolov5s_e_eiou')
# axs[1, 2].set_title('mAP_05')
# axs[1, 2].legend()
#
# axs[1, 3].plot(x,mAP_05_095_1,label='yolov5s_e_aeiou')
# axs[1, 3].plot(x,mAP_05_095_2,label='yolov5s')
# axs[1, 3].plot(x,mAP_05_095_3,label='yolov5s_e_eiou')
# axs[1, 3].set_title('mAP_05_095')
# axs[1, 3].legend()


