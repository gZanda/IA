import multiprocessing
import subprocess

def run_script(script_name, k):
    subprocess.run(["python", script_name, str(k)])

def main():

    # Get the value of k from user input or any other source
    k = int(input("Enter the number of clusters (k): "))
    script1 = "Hardcore.py"
    script2 = "Sklearn.py"

    p1 = multiprocessing.Process(target=run_script, args=(script1, k))
    p2 = multiprocessing.Process(target=run_script, args=(script2, k))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

if __name__ == "__main__":
    main()
