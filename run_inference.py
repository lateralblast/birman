import os
import sys
import signal
import platform
import argparse
import subprocess

def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

BITNET_2B_CHAT_TEMPLATE = os.path.join("chat-templates", "bitnet-b1.58-2B-4T.jinja")

def run_inference():
    build_dir = "build"
    # Current llama-cli is chat-only; plain completion needs llama-completion.
    tool = "llama-cli" if args.conversation else "llama-completion"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", f"{tool}.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", tool)
    else:
        main_path = os.path.join(build_dir, "bin", tool)
    command = [
        f'{main_path}',
        '-m', args.model,
        '-n', str(args.n_predict),
        '-t', str(args.threads),
        '-ngl', '0',
        '-c', str(args.ctx_size),
        '--temp', str(args.temperature),
    ]
    if args.conversation:
        # -p is the first user message in llama-cli; the README semantics are a system prompt.
        command += ['-sys', args.prompt, '-cnv']
        template = args.chat_template_file
        if template is None and "2B-4T" in args.model:
            # The GGUF's embedded template ends the prompt with an EOS token, so the model
            # answers with unrelated text. Use a template without it, and stop on "Human:"
            # because the model does not reliably emit EOS at the end of a turn.
            template = BITNET_2B_CHAT_TEMPLATE
            command += ['-r', 'Human:']
        if template:
            command += ['--chat-template-file', template]
    else:
        command += ['-p', args.prompt, '-no-cnv']
    run_command(command)

def signal_handler(sig, frame):
    print("Ctrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    # Usage: python run_inference.py -p "Microsoft Corporation is an American multinational corporation and technology company headquartered in Redmond, Washington."
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=False, default="models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict when generating text", required=False, default=128)
    parser.add_argument("-p", "--prompt", type=str, help="Prompt to generate text from", required=True)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=2)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the prompt context", required=False, default=2048)
    parser.add_argument("-temp", "--temperature", type=float, help="Temperature, a hyperparameter that controls the randomness of the generated text", required=False, default=0.8)
    parser.add_argument("-cnv", "--conversation", action='store_true', help="Whether to enable chat mode or not (for instruct models.)")
    parser.add_argument("--chat-template-file", type=str, help="Jinja chat template to use in chat mode (default: built in for BitNet-b1.58-2B-4T, otherwise the model's own)", required=False, default=None)

    args = parser.parse_args()
    run_inference()