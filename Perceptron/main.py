import multiprocessing
import subprocess

def run_script(script_name, k):
    subprocess.run(["python", script_name, str(k)])

def main():

    script1 = "mlp.py"
    script2 = "knn.py"

    p1 = multiprocessing.Process(target=run_script, args=(script1))
    p2 = multiprocessing.Process(target=run_script, args=(script2))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

if __name__ == "__main__":
    main()