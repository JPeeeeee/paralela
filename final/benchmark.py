#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import math
import random

ALPHABET = "ATGC"

def generate_sequences(size_k):
    """Gera arquivos fileA.in e fileB.in aleatórios de tamanho k."""
    seqA = "".join(random.choices(ALPHABET, k=size_k))
    seqB = "".join(random.choices(ALPHABET, k=size_k))
    with open("fileA.in", "w") as fp:
        fp.write(seqA)
    with open("fileB.in", "w") as fp:
        fp.write(seqB)
    print(f"[Setup] Geradas sequências com N = {size_k} caracteres em fileA.in e fileB.in")

def parse_output(stdout_text):
    """Extrai Score e Time da saída do programa."""
    score = None
    exec_time = None
    for line in stdout_text.splitlines():
        line = line.strip()
        if line.startswith("Score:"):
            try:
                score = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("Time:"):
            try:
                exec_time = float(line.split(":")[1].strip())
            except ValueError:
                pass
    return score, exec_time

def main():
    parser = argparse.ArgumentParser(
        description="Script de benchmark para executar o LCS paralelo 20x e calcular média e desvio padrão."
    )
    parser.add_argument("--runs", type=int, default=20, help="Número de execuções (padrão: 20)")
    parser.add_argument("--np", type=int, default=2, help="Número de processos MPI (-np)")
    parser.add_argument("--threads", type=int, default=2, help="Número de threads OpenMP por processo (OMP_NUM_THREADS)")
    parser.add_argument("--hostfile", type=str, default=None, help="Caminho do arquivo hostfile (opcional, para múltiplas máquinas)")
    parser.add_argument("--k", type=int, default=None, help="Tamanho das sequências N a gerar (se omitido, usa os arquivos existentes)")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Arquivo CSV de saída")

    args = parser.parse_args()

    # Se informado --k, gera novas sequências
    if args.k is not None:
        generate_sequences(args.k)
    else:
        if not (os.path.exists("fileA.in") and os.path.exists("fileB.in")):
            print("[Erro] Arquivos fileA.in e fileB.in não encontrados. Use --k <tamanho> para gerá-los.")
            sys.exit(1)

    # Monta o comando MPI
    cmd = ["mpirun"]
    if args.hostfile:
        cmd.extend(["--hostfile", args.hostfile])
    cmd.extend(["-np", str(args.np)])
    cmd.extend(["-x", f"OMP_NUM_THREADS={args.threads}"])
    cmd.append("./lcs_parallel")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)

    print("=" * 65)
    print(f"  BENCHMARK LCS HÍBRIDO (MPI + OpenMP)")
    print(f"  Comando      : {' '.join(cmd)}")
    print(f"  Processos MPI: {args.np}")
    print(f"  Threads OMP  : {args.threads}")
    print(f"  Repetições   : {args.runs}")
    print(f"  Arquivo CSV  : {args.output}")
    print("=" * 65)

    times = []
    scores = []

    for i in range(1, args.runs + 1):
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                env=env
            )
            score, t = parse_output(result.stdout)
            if t is None:
                print(f"[{i:02d}/{args.runs:02d}] Falha ao ler tempo! Saída recebida:\n{result.stdout}")
                continue
            
            times.append(t)
            scores.append(score)
            print(f"  [{i:02d}/{args.runs:02d}] Score: {score} | Tempo: {t:.6f} s")
        except subprocess.CalledProcessError as e:
            print(f"[{i:02d}/{args.runs:02d}] Erro na execução: {e.stderr}")
            sys.exit(1)

    if not times:
        print("[Erro] Nenhuma execução foi completada com sucesso.")
        sys.exit(1)

    # Estatísticas
    n = len(times)
    mean_time = sum(times) / n
    variance = sum((t - mean_time) ** 2 for t in times) / (n - 1) if n > 1 else 0.0
    std_dev = math.sqrt(variance)
    min_time = min(times)
    max_time = max(times)

    # Verifica se os scores foram idênticos em todas as execuções (corretude)
    consistent_score = len(set(scores)) == 1

    print("=" * 65)
    print("  RESULTADOS ESTATÍSTICOS")
    print("=" * 65)
    print(f"  Execuções com sucesso : {n}/{args.runs}")
    print(f"  Score constante       : {'SIM (' + str(scores[0]) + ')' if consistent_score else 'NÃO (divergente!)'}")
    print(f"  Tempo Mínimo          : {min_time:.6f} s")
    print(f"  Tempo Máximo          : {max_time:.6f} s")
    print(f"  Tempo Médio (mu)      : {mean_time:.6f} s")
    print(f"  Desvio Padrão (sigma) : {std_dev:.6f} s ({(std_dev / mean_time * 100):.2f}%)")
    print("=" * 65)

    # Salva no arquivo CSV
    with open(args.output, "w") as fp:
        fp.write("run,mpi_processes,omp_threads,score,time_seconds\n")
        for idx, (s, t) in enumerate(zip(scores, times), 1):
            fp.write(f"{idx},{args.np},{args.threads},{s},{t:.6f}\n")
        fp.write("\n")
        fp.write(f"# media,{mean_time:.6f}\n")
        fp.write(f"# desvio_padrao,{std_dev:.6f}\n")
        fp.write(f"# min,{min_time:.6f}\n")
        fp.write(f"# max,{max_time:.6f}\n")

    print(f"[Sucesso] Dados salvos com sucesso em: {args.output}\n")

if __name__ == "__main__":
    main()

