import cProfile
import subprocess
import sys

def run_streamlit_app():
    subprocess.run([sys.executable,"streamlit", "run", "main.py"])

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_streamlit_app()
    
    profiler.disable()
    profiler.dump_stats("profile_results.prof")