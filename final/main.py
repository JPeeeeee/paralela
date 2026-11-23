import random
import argparse


ALPHABET="ATGC"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=4)
    args = parser.parse_args()
        
    seqA = "".join(random.choices(ALPHABET, k=int(args.k)))
    seqB = "".join(random.choices(ALPHABET, k=int(args.k)))

    with open("fileA.in", "w") as fp:
        fp.write(seqA)

    with open("fileB.in", "w") as fp:
        fp.write(seqB)


if __name__ == "__main__":
    main()