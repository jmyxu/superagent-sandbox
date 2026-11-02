"""Minimal demo app for the dsh-superagent sandbox."""

import argparse

VERSION = "1.0.0"


def greet(name="world"):
    return f"Hello, {name}!"


def main():
    print(f"Version: {VERSION}")
    parser = argparse.ArgumentParser(description="Print a greeting.")
    parser.add_argument("--name", default="world", help="name to greet")
    args = parser.parse_args()
    print(greet(args.name))


if __name__ == "__main__":
    main()

