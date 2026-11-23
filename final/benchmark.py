#!/usr/bin/env python3
"""
benchmark.py - Script oficial de benchmark para a disciplina CI1316 (UFPR)
Professor: Marco Antonio Zanata Alves
Autores: Edison Luiz Matias Junior & João Pedro Vieira Santos

Executa a matriz completa de experimentos prevista no enunciado:
- Variação do tamanho de entrada: N = 10k, 20k, 40k
- Variação de trabalhadores: 1, 2, 4, 8 threads/processos
- 20 execuções por configuração para cálculo de média e desvio padrão
- Verificação estrita de corretude contra o sequencial
- Geração das tabelas completas de Speedup e Eficiência
- Atualização automática do relatório LaTeX (relatorio.tex)
"""

import os
import sys
import subprocess
import argparse
import math
import random
import time
import re

ALPHABET = "ATGC"

def generate_sequences(size_n, file_a="fileA.in", file_b="fileB.in"):
    """Gera arquivos de entrada com tamanho N e quebra de linha final."""
    seq_a = "".join(random.choices(ALPHABET, k=size_n))
    seq_b = "".join(random.choices(ALPHABET, k=size_n))
    with open(file_a, "w") as fa:
        fa.write(seq_a + "\n")
    with open(file_b, "w") as fb:
        fb.write(seq_b + "\n")

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

def run_command(cmd, env=None):
    """Executa comando medindo o tempo decorrido."""
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=proc_env)
    t1 = time.time()

    if res.returncode != 0:
        print(f"[Erro] Falha ao executar {cmd}:\n{res.stderr}", file=sys.stderr)
        return None, None
    
    score, measured_time = parse_output(res.stdout)
    if measured_time is None:
        measured_time = t1 - t0
    
    return score, measured_time

def mean_std(values):
    """Calcula média aritmética e desvio padrão amostral."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return mean, std

def update_latex_report(results, sizes, workers, report_path="relatorio.tex"):
    """Atualiza as tabelas de Speedup e Eficiência no relatorio.tex."""
    if not os.path.exists(report_path):
        return

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Monta Tabela de Speedup
    col_headers = " & ".join([f"\\textbf{{{w} Workers}}" for w in workers])
    sp_rows = []
    for s in sizes:
        base_t = results[s][1]["mean"]
        row_cells = []
        for w in workers:
            m = results[s][w]["mean"]
            sd = results[s][w]["std"]
            sp = base_t / m if m > 0 else 0.0
            row_cells.append(f"{sp:.2f}$\\times$ ($\\pm${sd:.3f}s)")
        sp_rows.append(f"$N = {s//1000}\\text{{k}}$ & " + " & ".join(row_cells) + " \\\\")

    speedup_table = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Tabela de Speedup ($S_p = T_1 / T_p$) com média e desvio padrão.}\n"
        "\\label{tab:speedup}\n"
        "\\begin{tabular}{@{}l" + "c" * len(workers) + "@{}}\n\\toprule\n"
        "\\textbf{Entrada} & " + col_headers + " \\\\ \\midrule\n"
        + "\n".join(sp_rows) + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}"
    )

    # Monta Tabela de Eficiência
    eff_rows = []
    for s in sizes:
        base_t = results[s][1]["mean"]
        row_cells = []
        for w in workers:
            m = results[s][w]["mean"]
            sp = base_t / m if m > 0 else 0.0
            eff = sp / w
            row_cells.append(f"{eff:.2f}")
        eff_rows.append(f"$N = {s//1000}\\text{{k}}$ & " + " & ".join(row_cells) + " \\\\")

    efficiency_table = (
        "\\begin{table}[h!]\n\\centering\n"
        "\\caption{Tabela de Eficiência ($E_p = S_p / p$) correspondente.}\n"
        "\\label{tab:eficiencia}\n"
        "\\begin{tabular}{@{}l" + "c" * len(workers) + "@{}}\n\\toprule\n"
        "\\textbf{Entrada} & " + col_headers + " \\\\ \\midrule\n"
        + "\n".join(eff_rows) + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}"
    )

    # Substitui tabelas existentes usando lambda para evitar escape de regex
    content = re.sub(r"\\begin\{table\}\[h!\].*?\\label\{tab:speedup\}.*?\\end\{table\}", lambda m: speedup_table, content, flags=re.DOTALL)
    content = re.sub(r"\\begin\{table\}\[h!\].*?\\label\{tab:eficiencia\}.*?\\end\{table\}", lambda m: efficiency_table, content, flags=re.DOTALL)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[Sucesso] Relatório '{report_path}' atualizado automaticamente com as novas tabelas!")

def main():
    parser = argparse.ArgumentParser(description="Automação experimental do LCS (OpenMP + MPI) - CI1316")
    parser.add_argument("--runs", type=int, default=20, help="Número de execuções por teste (padrão: 20)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000, 20000, 40000], help="Tamanhos de N (padrão: 10000 20000 40000)")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8], help="Trabalhadores paralelos (padrão: 1 2 4 8)")
    parser.add_argument("--k", type=int, default=100, help="Tamanho do bloco K (padrão: 100)")
    parser.add_argument("--hostfile", type=str, default=None, help="Caminho do arquivo hostfile para o DINF (ex: hosts)")
    parser.add_argument("--mode", choices=["omp", "mpi", "hybrid"], default="omp", help="Modo de paralelização dos workers")
    parser.add_argument("--quick", action="store_true", help="Modo rápido de teste (tamanhos menores e 3 repetições)")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Arquivo CSV de saída")
    args = parser.parse_args()

    if args.quick:
        args.runs = 3
        args.sizes = [2000, 5000]
        args.workers = [1, 2, 4]
        args.k = 50

    print("=" * 70)
    print("  BENCHMARK LCS OFICIAL - PROGRAMAÇÃO PARALELA (CI1316 UFPR)")
    print(f"  Tamanhos de Entrada (N) : {args.sizes}")
    print(f"  Trabalhadores           : {args.workers}")
    print(f"  Tamanho do Bloco (K)    : {args.k}")
    print(f"  Modo                    : {args.mode.upper()}")
    print(f"  Repetições por Ponto    : {args.runs} execuções (conforme enunciado)")
    print(f"  Arquivo CSV             : {args.output}")
    print("=" * 70)

    # 1. Compilar os binários
    print("\n[1/3] Compilando os binários com otimização máxima (-O3)...")
    res = subprocess.run(["make", "all"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[Erro] Falha na compilação:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)
    print("Compilação concluída: 'lcs_parallel' e 'lcs_seq' prontos.\n")

    # Estrutura de dados: {size: {worker: {"mean": float, "std": float, "times": []}}}
    results = {s: {} for s in args.sizes}
    csv_lines = ["run,size_n,workers,mode,score,time_seconds\n"]

    # 2. Executar experimentos
    for size in args.sizes:
        print("=" * 70)
        print(f"  INICIANDO TESTES PARA N = {size} ({size//1000}k)")
        print("=" * 70)

        generate_sequences(size)

        # Baseline sequencial
        print(f"-> Medindo baseline sequencial ({args.runs} execuções)...")
        seq_times = []
        expected_score = None
        for r in range(args.runs):
            s, t = run_command(["./lcs_seq"])
            if t is not None:
                seq_times.append(t)
                if expected_score is None:
                    expected_score = s
        seq_mean, seq_std = mean_std(seq_times)
        print(f"   [Sequencial Base] Tempo: {seq_mean:.4f}s (±{seq_std:.4f}s) | Score: {expected_score}")

        for w in args.workers:
            print(f"\n-> Testando com {w} trabalhadores ({args.mode.upper()}) [{args.runs} repetições]...")
            t_list = []
            scores = []
            
            for r in range(1, args.runs + 1):
                env = os.environ.copy()
                env["LCS_BLOCK_SIZE"] = str(args.k)

                if args.mode == "omp":
                    cmd = ["./lcs_parallel"]
                    env["OMP_NUM_THREADS"] = str(w)
                elif args.mode == "mpi":
                    cmd = ["mpirun"]
                    if args.hostfile:
                        cmd.extend(["--hostfile", args.hostfile])
                    cmd.extend(["-np", str(w)])
                    cmd.extend(["-x", "OMP_NUM_THREADS=1"])
                    cmd.extend(["-x", f"LCS_BLOCK_SIZE={args.k}"])
                    cmd.append("./lcs_parallel")
                    env["OMP_NUM_THREADS"] = "1"
                elif args.mode == "hybrid":
                    np = 2 if w >= 2 else 1
                    nth = max(1, w // np)
                    cmd = ["mpirun"]
                    if args.hostfile:
                        cmd.extend(["--hostfile", args.hostfile])
                    cmd.extend(["-np", str(np)])
                    cmd.extend(["-x", f"OMP_NUM_THREADS={nth}"])
                    cmd.extend(["-x", f"LCS_BLOCK_SIZE={args.k}"])
                    cmd.append("./lcs_parallel")
                    env["OMP_NUM_THREADS"] = str(nth)

                score, t = run_command(cmd, env=env)
                if t is not None:
                    t_list.append(t)
                    scores.append(score)
                    csv_lines.append(f"{r},{size},{w},{args.mode},{score},{t:.6f}\n")
                    print(f"   [{r:02d}/{args.runs:02d}] Tempo: {t:.4f}s | Score: {score}")

            w_mean, w_std = mean_std(t_list)
            consistent = (len(set(scores)) == 1) and (scores[0] == expected_score)
            status_str = "CORRETO" if consistent else f"DIVERGIU (esperado {expected_score})"
            results[size][w] = {"mean": w_mean, "std": w_std, "times": t_list}
            print(f"   Resultado ({w} trab.): Média = {w_mean:.4f}s (±{w_std:.4f}s) | Status: {status_str}")

    # Salva CSV
    with open(args.output, "w") as fp:
        fp.writelines(csv_lines)
    print(f"\n[2/3] Resultados brutos salvos com sucesso em: {args.output}")

    # 3. Imprime Tabelas Formatadas (Item 6)
    print("\n" + "=" * 70)
    print("  TABELAS DE SPEEDUP E EFICIÊNCIA GERADAS (ITEM 6 DO ENUNCIADO)")
    print("=" * 70)

    # Tabela Speedup
    header_w = " | ".join([f"{w:2d} Trab." for w in args.workers])
    print(f"\n--- TABELA DE SPEEDUP (Sp = T1 / Tp) ---")
    print(f"| Entrada | {header_w} |")
    print(f"| :--- | " + " | ".join([":---:" for _ in args.workers]) + " |")
    for s in args.sizes:
        base_t = results[s][1]["mean"]
        row = [f"N={s//1000}k"]
        for w in args.workers:
            m = results[s][w]["mean"]
            sd = results[s][w]["std"]
            sp = base_t / m if m > 0 else 0.0
            row.append(f"{sp:.2f}x (±{sd:.3f})")
        print("| " + " | ".join(row) + " |")

    # Tabela Eficiência
    print(f"\n--- TABELA DE EFICIÊNCIA (Ep = Sp / p) ---")
    print(f"| Entrada | {header_w} |")
    print(f"| :--- | " + " | ".join([":---:" for _ in args.workers]) + " |")
    for s in args.sizes:
        base_t = results[s][1]["mean"]
        row = [f"N={s//1000}k"]
        for w in args.workers:
            m = results[s][w]["mean"]
            sp = base_t / m if m > 0 else 0.0
            eff = sp / w
            row.append(f"{eff:.2f}")
        print("| " + " | ".join(row) + " |")

    # Atualiza relatório LaTeX
    update_latex_report(results, args.sizes, args.workers)

if __name__ == "__main__":
    main()
