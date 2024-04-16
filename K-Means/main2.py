import multiprocessing
import subprocess
from time import sleep

def run_script(script_name):
    subprocess.run(["python", script_name])

def main():
    script1 = "Hardcore.py"
    script2 = "Sklearn.py"

    p1 = multiprocessing.Process(target=run_script, args=(script1,))
    p2 = multiprocessing.Process(target=run_script, args=(script2,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

if __name__ == "__main__":
    main()
