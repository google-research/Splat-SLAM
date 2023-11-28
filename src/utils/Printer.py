# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from colorama import Fore, Style
import torch.multiprocessing as mp


class FontColor(object):
    MAPPER=Fore.CYAN
    TRACKER=Fore.BLUE
    INFO=Fore.YELLOW
    ERROR=Fore.RED
    PCL=Fore.GREEN
    EVAL=Fore.MAGENTA
    MESH="yellow"


def get_msg_prefix(color):
    if color == FontColor.MAPPER:
        msg_prefix = color + "[MAPPER] " + Style.RESET_ALL
    elif color ==  FontColor.TRACKER:
        msg_prefix = color + "[TRACKER] " + Style.RESET_ALL
    elif color ==  FontColor.INFO:
        msg_prefix = color + "[INFO] " + Style.RESET_ALL
    elif color ==  FontColor.ERROR:
        msg_prefix = color + "[ERROR] " + Style.RESET_ALL
    elif color ==  FontColor.PCL:
        msg_prefix = color + "[POINTCLOUD] " + Style.RESET_ALL
    elif color ==  FontColor.EVAL:
        msg_prefix = color + "[EVALUATION] " + Style.RESET_ALL
    elif color == FontColor.MESH:
        msg_prefix = FontColor.INFO + "[MESH] " + Style.RESET_ALL
    else:
        msg_prefix = Style.RESET_ALL
    return msg_prefix

class TrivialPrinter(object):
    def print(self,msg:str,color=None):
        msg_prefix = get_msg_prefix(color)
        msg = msg_prefix + msg + Style.RESET_ALL
        print(msg)        

class Printer(TrivialPrinter):
    def __init__(self, total_img_num):
        self.msg_lock = mp.Lock()
        self.msg_queue = mp.Queue()
        self.progress_counter = mp.Value('i', 0)
        process = mp.Process(target=self.printer_process, args=(total_img_num,))
        process.start()
    def print(self,msg:str,color=None):
        msg_prefix = get_msg_prefix(color)
        msg = msg_prefix + msg + Style.RESET_ALL
        with self.msg_lock:
            self.msg_queue.put(msg)
    def update_pbar(self):
        with self.msg_lock:
            self.progress_counter.value += 1
            self.msg_queue.put(f"PROGRESS")
    def pbar_ready(self):
        with self.msg_lock:
            self.msg_queue.put(f"READY")        

    def printer_process(self,total_img_num):
        from tqdm import tqdm
        while True:
            message = self.msg_queue.get()
            if message == "READY":
                break
            else:
                print(message)
        with tqdm(total=total_img_num) as pbar:
            while self.progress_counter.value < total_img_num:
                message = self.msg_queue.get()
                if message == "DONE":
                    break
                elif message.startswith("PROGRESS"):
                    with self.msg_lock:
                        completed = self.progress_counter.value
                    pbar.set_description(FontColor.TRACKER+f"[TRACKER] "+Style.RESET_ALL)
                    pbar.n = completed
                    pbar.refresh()
                else:
                    pbar.write(message)
        while True:
            message = self.msg_queue.get()
            if message == "DONE":
                break
            else:
                print(message)
            
    
    def terminate(self):
        self.msg_queue.put("DONE")


