#!/usr/bin/env python3
"""
benchmark.py - Script de automação experimental para LCS (OpenMP + MPI)
Disciplina CI1316 - Programação Paralela - UFPR
Prof. Marco Antonio Zanata Alves

Gera os resultados em uma pasta separada para facilitar a elaboração do relatório:
- hardware_env.txt  : Especificações da máquina (SO, Kernel, CPU, Cores, Compilador)
- dados_relatorio.md: Tabelas prontas em Markdown para os Itens 3, 4, 5 e 6
- tabelas.tex       : Tabelas em formato LaTeX para Overleaf/LaTeX
- resultados.csv    : Planilha CSV com dados brutos, médias e desvios padrão
- dados.json        : Dados brutos em formato JSON
- Gráficos PNG      : Gráficos de Speedup e Eficiência prontos para o relatório
"""

import os
import sys
import time
import math
import random
import string
import argparse
import subprocess
import json
import csv
import platform

def get_system_specs():
    """Captura especificações do sistema e compilador para o Item 3 do relatório."""
    specs = {}
    specs["os"] = platform.system()
    specs["os_release"] = platform.release()
    specs["os_version"] = platform.version()
    specs["machine"] = platform.machine()
    
    # CPU Model & Cores
    cpu_model = "Desconhecido"
    p_cores = "N/A"
    e_cores = "N/A"
    total_cores = os.cpu_count() or 1

    if specs["os"] == "Darwin":
        try:
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], stdout=subprocess.PIPE, text=True)
            if res.returncode == 0 and res.stdout.strip():
                cpu_model = res.stdout.strip()
            
            p_res = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"], stdout=subprocess.PIPE, text=True)
            if p_res.returncode == 0:
                p_cores = p_res.stdout.strip()

            e_res = subprocess.run(["sysctl", "-n", "hw.perflevel1.logicalcpu"], stdout=subprocess.PIPE, text=True)
            if e_res.returncode == 0:
                e_cores = e_res.stdout.strip()
        except Exception:
            pass
    elif specs["os"] == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":")[1].strip()
                        break
        except Exception:
            pass

    specs["cpu_model"] = cpu_model
    specs["total_cores"] = total_cores
    specs["p_cores"] = p_cores
    specs["e_cores"] = e_cores

    # Compilador
    gcc_ver = "Desconhecido"
    try:
        res = subprocess.run(["/opt/homebrew/bin/gcc-16", "--version"], stdout=subprocess.PIPE, text=True)
        if res.returncode == 0:
            gcc_ver = res.stdout.splitlines()[0]
    except Exception:
        try:
            res = subprocess.run(["gcc", "--version"], stdout=subprocess.PIPE, text=True)
            if res.returncode == 0:
                gcc_ver = res.stdout.splitlines()[0]
        except Exception:
            pass
    specs["compiler"] = gcc_ver

    # OpenMPI
    mpi_ver = "Desconhecido"
    try:
        res = subprocess.run(["mpirun", "--version"], stdout=subprocess.PIPE, text=True)
        if res.returncode == 0:
            mpi_ver = res.stdout.splitlines()[0]
    except Exception:
        pass
    specs["mpi_version"] = mpi_ver

    return specs

def generate_sequences(size_a, size_b, file_a="fileA.in", file_b="fileB.in"):
    """Gera sequências aleatórias (letras e números) e salva nos arquivos de entrada."""
    chars = string.ascii_letters + string.digits
    seq_a = "".join(random.choices(chars, k=size_a))
    seq_b = "".join(random.choices(chars, k=size_b))
    with open(file_a, "w") as fa:
        fa.write(seq_a + "\n")
    with open(file_b, "w") as fb:
        fb.write(seq_b + "\n")

def run_command(cmd, env=None):
    """Executa um comando e extrai métricas de tempo e score."""
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    
    start_wall = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=proc_env)
    end_wall = time.time()

    if res.returncode != 0:
        print(f"Erro ao executar {cmd}:\n{res.stderr}", file=sys.stderr)
        return None

    score = None
    time_total = end_wall - start_wall
    time_seq = 0.0
    time_comp = time_total

    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("Score:"):
            try:
                score = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("TIME_TOTAL:"):
            time_total = float(line.split(":")[1].strip())
        elif line.startswith("TIME_SEQ:"):
            time_seq = float(line.split(":")[1].strip())
        elif line.startswith("TIME_COMP:"):
            time_comp = float(line.split(":")[1].strip())

    return {
        "score": score,
        "time_total": time_total,
        "time_seq": time_seq,
        "time_comp": time_comp
    }

def mean_std(values):
    """Calcula média e desvio padrão amostral."""
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

def plot_charts(perf_data, sizes, outdir):
    """Gera gráficos de Speedup e Eficiência com Matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib não encontrado. Pulando geração automática de gráficos PNG.")
        return

    # Gráfico de Speedup
    for mode in ["openmp", "mpi"]:
        if mode not in perf_data or not perf_data[mode]:
            continue
        
        plt.figure(figsize=(8, 5))
        label_axis = "Número de Threads" if mode == "openmp" else "Número de Processos"
        title_mode = "OpenMP" if mode == "openmp" else "MPI"

        all_p = sorted(list(next(iter(perf_data[mode].values())).keys()))

        # Linha ideal linear
        plt.plot(all_p, all_p, "k--", label="Speedup Linear (Ideal)", alpha=0.7)

        for size in sizes:
            base_t, _ = perf_data[mode][size][1]
            speedups = []
            for p in all_p:
                m_t, _ = perf_data[mode][size][p]
                speedups.append(base_t / m_t if m_t > 0 else 0)
            plt.plot(all_p, speedups, marker="o", label=f"N = {size}")

        plt.xlabel(label_axis, fontsize=12)
        plt.ylabel("Speedup", fontsize=12)
        plt.title(f"Speedup vs {label_axis} ({title_mode})", fontsize=14)
        plt.xticks(all_p)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"grafico_speedup_{mode}.png"), dpi=300)
        plt.close()

        # Gráfico de Eficiência
        plt.figure(figsize=(8, 5))
        plt.axhline(1.0, color="k", linestyle="--", label="Eficiência Ideal (1.0)", alpha=0.7)

        for size in sizes:
            base_t, _ = perf_data[mode][size][1]
            efficiencies = []
            for p in all_p:
                m_t, _ = perf_data[mode][size][p]
                sp = base_t / m_t if m_t > 0 else 0
                efficiencies.append(sp / p)
            plt.plot(all_p, efficiencies, marker="s", label=f"N = {size}")

        plt.xlabel(label_axis, fontsize=12)
        plt.ylabel("Eficiência", fontsize=12)
        plt.title(f"Eficiência vs {label_axis} ({title_mode})", fontsize=14)
        plt.xticks(all_p)
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"grafico_eficiencia_{mode}.png"), dpi=300)
        plt.close()

    print(f"Gráficos PNG salvos em '{outdir}'.")

def export_latex_tables(amdahl_data, perf_data, sizes, threads, procs, outdir):
    """Gera tabelas prontas em formato LaTeX para o relatório."""
    tex_path = os.path.join(outdir, "tabelas.tex")
    procs_amdahl = [1, 2, 4, 8, 16, 32, 64]

    with open(tex_path, "w") as f:
        f.write("% ======================================================================\n")
        f.write("% TABELAS GERADAS AUTOMATICAMENTE PARA O RELATÓRIO CI1316 (UFPR)\n")
        f.write("% ======================================================================\n\n")

        # Tabela Item 4
        f.write("% --- TABELA ITEM 4: Região Sequencial vs Paralelizável ---\n")
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{Análise da Região Sequencial e Paralelizável (Item 4)}\n")
        f.write("\\label{tab:item4}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n")
        f.write("Tamanho ($N$) & Tempo Total (s) & Região Serial (s) & Fração Serial ($S$) & Fração Paralelizável ($P$) \\\\ \\hline\n")
        for size, d in amdahl_data.items():
            f.write(f"$N={size}$ & {d['T_total']:.4f} & {d['T_seq']:.4f} ({d['f_seq_pct']:.1f}\\%) & {d['S']:.4f} & {d['P']:.4f} ({d['f_par_pct']:.1f}\\%) \\\\ \\hline\n")
        f.write("\\end{tabular}\n\\end{table}\n\n")

        # Tabela Item 5
        f.write("% --- TABELA ITEM 5: Lei de Amdahl ---\n")
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{Speedup Máximo Teórico segundo a Lei de Amdahl (Item 5)}\n")
        f.write("\\label{tab:item5}\n")
        cols_spec = "|c|" + "c|" * len(procs_amdahl) + "c|"
        f.write(f"\\begin{{tabular}}{{{cols_spec}}}\n\\hline\n")
        f.write("Tamanho & " + " & ".join(f"$P={p}$" for p in procs_amdahl) + " & $P=\\infty$ \\\\ \\hline\n")
        for size, d in amdahl_data.items():
            P = d["P"]
            vals = [f"{1.0 / ((1.0 - P) + (P / p)):.2f}$\\times$" for p in procs_amdahl]
            inf_val = f"{1.0 / (1.0 - P):.2f}$\\times$" if (1.0 - P) > 0 else "$\\infty$"
            f.write(f"$N={size}$ & " + " & ".join(vals) + f" & {inf_val} \\\\ \\hline\n")
        f.write("\\end{tabular}\n\\end{table}\n\n")

        # Tabelas Item 6
        for mode in ["openmp", "mpi"]:
            if mode not in perf_data or not perf_data[mode]:
                continue
            title = "OpenMP" if mode == "openmp" else "MPI"
            units = "Threads" if mode == "openmp" else "Processos"
            params = threads if mode == "openmp" else procs

            # Speedup
            f.write(f"% --- TABELA ITEM 6: Speedup {title} ---\n")
            f.write("\\begin{table}[h!]\n\\centering\n")
            f.write(f"\\caption{{Speedup Empírico ({title}) com Média e Desvio Padrão}}\n")
            f.write(f"\\label{{tab:speedup_{mode}}}\n")
            cols = "|c|" + "c|" * len(params)
            f.write(f"\\begin{{tabular}}{{{cols}}}\n\\hline\n")
            f.write("Tamanho & " + " & ".join(f"{p} {units}" for p in params) + " \\\\ \\hline\n")
            for size in sizes:
                base_t, _ = perf_data[mode][size][1]
                row_vals = []
                for p in params:
                    m_t, s_t = perf_data[mode][size][p]
                    sp = base_t / m_t if m_t > 0 else 0
                    row_vals.append(f"{sp:.2f}$\\times$ ($\\pm${s_t:.3f})")
                f.write(f"$N={size}$ & " + " & ".join(row_vals) + " \\\\ \\hline\n")
            f.write("\\end{tabular}\n\\end{table}\n\n")

            # Eficiência
            f.write(f"% --- TABELA ITEM 6: Eficiência {title} ---\n")
            f.write("\\begin{table}[h!]\n\\centering\n")
            f.write(f"\\caption{{Eficiência Empírica ({title})}}\n")
            f.write(f"\\label{{tab:eficiencia_{mode}}}\n")
            f.write(f"\\begin{{tabular}}{{{cols}}}\n\\hline\n")
            f.write("Tamanho & " + " & ".join(f"{p} {units}" for p in params) + " \\\\ \\hline\n")
            for size in sizes:
                base_t, _ = perf_data[mode][size][1]
                row_vals = []
                for p in params:
                    m_t, _ = perf_data[mode][size][p]
                    sp = base_t / m_t if m_t > 0 else 0
                    eff = sp / p
                    row_vals.append(f"{eff:.2f}")
                f.write(f"$N={size}$ & " + " & ".join(row_vals) + " \\\\ \\hline\n")
            f.write("\\end{tabular}\n\\end{table}\n\n")

    print(f"Tabelas LaTeX salvas em '{tex_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Automação experimental do LCS (OpenMP + MPI)")
    parser.add_argument("--runs", type=int, default=20, help="Número de execuções por teste (padrão: 20)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000, 20000, 40000], help="Tamanhos de N a testar")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8], help="Variação de threads OpenMP")
    parser.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8], help="Variação de processos MPI")
    parser.add_argument("--mode", choices=["openmp", "mpi", "both"], default="both", help="Modo de paralelização a testar")
    parser.add_argument("--outdir", type=str, default="resultados_benchmark", help="Diretório onde salvar todos os resultados")
    parser.add_argument("--quick", action="store_true", help="Modo rápido de teste (tamanhos menores e 3 repetições)")
    args = parser.parse_args()

    if args.quick:
        args.runs = 3
        args.sizes = [2000, 4000]
        args.threads = [1, 2, 4]
        args.procs = [1, 2, 4]

    # Cria diretório de saída
    os.makedirs(args.outdir, exist_ok=True)

    print("======================================================================")
    print("INICIANDO EXPERIMENTOS LCS - CI1316 PROGRAMAÇÃO PARALELA (UFPR)")
    print(f"Repetições por configuração : {args.runs}")
    print(f"Tamanhos de entrada (N)    : {args.sizes}")
    print(f"Threads OpenMP              : {args.threads}")
    print(f"Processos MPI               : {args.procs}")
    print(f"Modo                        : {args.mode}")
    print(f"Pasta de Saída              : {args.outdir}/")
    print("======================================================================")

    # 1. Coleta Informações do Ambiente (Item 3)
    specs = get_system_specs()
    env_file = os.path.join(args.outdir, "hardware_env.txt")
    with open(env_file, "w") as f_env:
        f_env.write("METADADOS DE AMBIENTE EXPERIMENTAL (ITEM 3 DO RELATÓRIO)\n")
        f_env.write("=======================================================\n")
        f_env.write(f"Data do Experimento : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_env.write(f"Sistema Operacional  : {specs['os']} ({specs['os_release']}, {specs['machine']})\n")
        f_env.write(f"Versão do SO / Kernel: {specs['os_version']}\n")
        f_env.write(f"Modelo do Processador: {specs['cpu_model']}\n")
        f_env.write(f"Núcleos Totais       : {specs['total_cores']}\n")
        f_env.write(f"Performance Cores    : {specs['p_cores']}\n")
        f_env.write(f"Efficiency Cores     : {specs['e_cores']}\n")
        f_env.write(f"Compilador C         : {specs['compiler']}\n")
        f_env.write(f"Flags de Compilação  : -O3 -Wall -fopenmp\n")
        f_env.write(f"Versão OpenMPI       : {specs['mpi_version']}\n")
        f_env.write(f"Número de Repetições : {args.runs} execuções por configuração\n")
    print(f"\n[1/4] Metadados de hardware e compilador salvos em '{env_file}'.")

    # 2. Compilação
    print("\n[2/4] Compilando os programas com otimização máxima (-O3)...")
    res = subprocess.run(["make", "all"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Falha na compilação:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)
    print("Compilação concluída com sucesso (lcs_seq e lcs_parallel prontos).")

    # Estruturas para armazenamento de resultados
    csv_rows = []
    csv_rows.append(["Modo", "Tamanho_N", "Threads_Procs", "Execucao", "Score", "Tempo_Total", "Tempo_Seq", "Tempo_Comp"])

    perf_data = {"openmp": {}, "mpi": {}}
    amdahl_data = {}

    # 3. Execução dos Benchmarks
    print("\n[3/4] Executando medições de tempo...")
    for size in args.sizes:
        print(f"\n======================================================================")
        print(f"-> EXECUTANDO EXPERIMENTOS PARA N = {size}")
        print(f"======================================================================")
        
        generate_sequences(size, size)

        # Baseline Sequencial
        print(f"Executando Baseline Sequencial ({args.runs}x)...")
        seq_times_total = []
        seq_times_seq = []
        seq_times_comp = []
        expected_score = None

        for r in range(args.runs):
            out = run_command(["./lcs_seq"])
            if out:
                seq_times_total.append(out["time_total"])
                seq_times_seq.append(out["time_seq"])
                seq_times_comp.append(out["time_comp"])
                if expected_score is None:
                    expected_score = out["score"]
                csv_rows.append(["Sequencial", size, 1, r + 1, out["score"], out["time_total"], out["time_seq"], out["time_comp"]])

        m_total, s_total = mean_std(seq_times_total)
        m_seq, s_seq = mean_std(seq_times_seq)
        m_comp, s_comp = mean_std(seq_times_comp)

        f_seq = (m_seq / m_total) * 100.0 if m_total > 0 else 0.0
        f_par = (m_comp / m_total) * 100.0 if m_total > 0 else 0.0

        amdahl_data[size] = {
            "T_total": m_total,
            "T_seq": m_seq,
            "T_comp": m_comp,
            "f_seq_pct": f_seq,
            "f_par_pct": f_par,
            "P": f_par / 100.0,
            "S": f_seq / 100.0
        }

        print(f"  [Sequencial] Tempo Total: {m_total:.4f}s (±{s_total:.4f}s)")
        print(f"  [Sequencial] Região Serial (I/O): {m_seq:.4f}s ({f_seq:.2f}%)")
        print(f"  [Sequencial] Região Paralelizável (LCS): {m_comp:.4f}s ({f_par:.2f}%)")
        print(f"  [Sequencial] Score obtido: {expected_score}")

        # Teste OpenMP
        if args.mode in ["openmp", "both"]:
            perf_data["openmp"][size] = {}
            print(f"\n--- Testes OpenMP (1 a {max(args.threads)} threads) ---")
            for t in args.threads:
                t_list = []
                correct = True
                for r in range(args.runs):
                    out = run_command(["./lcs_parallel"], env={"OMP_NUM_THREADS": str(t)})
                    if out:
                        t_list.append(out["time_total"])
                        if out["score"] != expected_score:
                            correct = False
                        csv_rows.append(["OpenMP", size, t, r + 1, out["score"], out["time_total"], out["time_seq"], out["time_comp"]])
                
                m_t, s_t = mean_std(t_list)
                perf_data["openmp"][size][t] = (m_t, s_t)
                status_str = "OK" if correct else f"DIVERGIU (esperado {expected_score})"
                print(f"  OpenMP ({t:2d} threads): {m_t:.4f}s (±{s_t:.4f}s) | Status: {status_str}")

        # Teste MPI
        if args.mode in ["mpi", "both"]:
            perf_data["mpi"][size] = {}
            print(f"\n--- Testes MPI (1 a {max(args.procs)} processos) ---")
            for p in args.procs:
                t_list = []
                correct = True
                for r in range(args.runs):
                    cmd = ["mpirun", "-np", str(p), "./lcs_parallel"]
                    out = run_command(cmd, env={"OMP_NUM_THREADS": "1"})
                    if out:
                        t_list.append(out["time_total"])
                        if out["score"] != expected_score:
                            correct = False
                        csv_rows.append(["MPI", size, p, r + 1, out["score"], out["time_total"], out["time_seq"], out["time_comp"]])
                
                m_p, s_p = mean_std(t_list)
                perf_data["mpi"][size][p] = (m_p, s_p)
                status_str = "OK" if correct else f"DIVERGIU (esperado {expected_score})"
                print(f"  MPI    ({p:2d} procs  ): {m_p:.4f}s (±{s_p:.4f}s) | Status: {status_str}")

    # Salva CSV com todas as execuções brutas
    csv_path = os.path.join(args.outdir, "resultados.csv")
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerows(csv_rows)
    print(f"\n[4/4] Planilha CSV bruta salva em '{csv_path}'.")

    # Exporta JSON
    json_path = os.path.join(args.outdir, "dados.json")
    with open(json_path, "w") as f_json:
        json.dump({"specs": specs, "amdahl": amdahl_data, "perf": perf_data}, f_json, indent=2)
    print(f"Dados estruturados JSON salvos em '{json_path}'.")

    # Exporta Tabelas em LaTeX
    export_latex_tables(amdahl_data, perf_data, args.sizes, args.threads, args.procs, args.outdir)

    # Gera Gráficos PNG
    plot_charts(perf_data, args.sizes, args.outdir)

    # Gera Relatório Consolidado em Markdown
    md_path = os.path.join(args.outdir, "dados_relatorio.md")
    procs_amdahl = [1, 2, 4, 8, 16, 32, 64, "inf"]
    with open(md_path, "w") as f_md:
        f_md.write("# Dados Experimentais e Tabelas para o Relatório\n\n")
        f_md.write(f"**Data da Execução:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f_md.write("## 1. Metodologia e Especificações de Ambiente (Item 3)\n\n")
        f_md.write(f"- **Processador:** {specs['cpu_model']}\n")
        f_md.write(f"- **Topologia:** {specs['total_cores']} núcleos no total ({specs['p_cores']} núcleos de performance, {specs['e_cores']} núcleos de eficiência)\n")
        f_md.write(f"- **Sistema Operacional:** {specs['os']} {specs['os_release']} ({specs['machine']})\n")
        f_md.write(f"- **Versão do Kernel:** {specs['os_version']}\n")
        f_md.write(f"- **Compilador:** `{specs['compiler']}` com flags `-O3 -Wall -fopenmp`\n")
        f_md.write(f"- **Ambiente MPI:** `{specs['mpi_version']}`\n")
        f_md.write(f"- **Repetições:** {args.runs} execuções independentes para cada ponto amostral\n\n")

        f_md.write("## 2. Análise da Região Sequencial vs. Paralelizável (Item 4)\n\n")
        f_md.write("| Tamanho (N) | Tempo Total Médio (s) | Região Serial I/O (s) | Fração Serial ($S$) | Fração Paralelizável ($P$) |\n")
        f_md.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for size, d in amdahl_data.items():
            f_md.write(f"| N={size} | {d['T_total']:.4f}s | {d['T_seq']:.4f}s ({d['f_seq_pct']:.2f}%) | {d['S']:.4f} | {d['P']:.4f} ({d['f_par_pct']:.2f}%) |\n")

        f_md.write("\n## 3. Speedup Máximo Teórico - Lei de Amdahl (Item 5)\n\n")
        f_md.write("| Tamanho (N) | " + " | ".join(f"P={p}" for p in procs_amdahl) + " |\n")
        f_md.write("| :--- | " + " | ".join(":---:" for _ in procs_amdahl) + " |\n")
        for size, d in amdahl_data.items():
            P = d["P"]
            row = [f"N={size}"]
            for p in procs_amdahl:
                if p == "inf":
                    sp = 1.0 / (1.0 - P) if (1.0 - P) > 0 else float("inf")
                else:
                    sp = 1.0 / ((1.0 - P) + (P / p))
                row.append(f"{sp:.2f}x")
            f_md.write("| " + " | ".join(row) + " |\n")

        sec_idx = 4
        for mode in ["openmp", "mpi"]:
            if mode not in perf_data or not perf_data[mode]:
                continue
            title = "OpenMP" if mode == "openmp" else "MPI"
            units = "Threads" if mode == "openmp" else "Processos"
            params = args.threads if mode == "openmp" else args.procs

            f_md.write(f"\n## {sec_idx}. Speedup Empírico - {title} (Item 6)\n\n")
            sec_idx += 1
            f_md.write(f"| Tamanho (N) | " + " | ".join(f"{p} {units}" for p in params) + " |\n")
            f_md.write("| :--- | " + " | ".join(":---:" for _ in params) + " |\n")
            for size in args.sizes:
                base_t, _ = perf_data[mode][size][1]
                row = [f"N={size}"]
                for p in params:
                    m_t, s_t = perf_data[mode][size][p]
                    sp = base_t / m_t if m_t > 0 else 0.0
                    row.append(f"{sp:.2f}x (±{s_t:.3f}s)")
                f_md.write("| " + " | ".join(row) + " |\n")

            f_md.write(f"\n## {sec_idx}. Eficiência Empírica - {title} (Item 6)\n\n")
            sec_idx += 1
            f_md.write(f"| Tamanho (N) | " + " | ".join(f"{p} {units}" for p in params) + " |\n")
            f_md.write("| :--- | " + " | ".join(":---:" for _ in params) + " |\n")
            for size in args.sizes:
                base_t, _ = perf_data[mode][size][1]
                row = [f"N={size}"]
                for p in params:
                    m_t, _ = perf_data[mode][size][p]
                    sp = base_t / m_t if m_t > 0 else 0.0
                    eff = sp / p
                    row.append(f"{eff:.2f}")
                f_md.write("| " + " | ".join(row) + " |\n")

    print(f"\nDocumento Markdown consolidado salvo em '{md_path}'.")
    print(f"Todos os artefatos prontos na pasta: '{args.outdir}/'.")

if __name__ == "__main__":
    main()
