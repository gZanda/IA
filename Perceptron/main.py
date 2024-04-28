import subprocess

def run_script(script):
    subprocess.run(['python', script])

if __name__ == "__main__":
    script1 = 'knn.py'
    script2 = 'mlp.py'

    run_script(script1)
    run_script(script2)
