"""Minimal demo app for the dsh-superagent sandbox."""

VERSION = "1.0.0"


def greet(name="world"):
    return f"Hello, {name}!"


def main():
    print(f"Version: {VERSION}")
    print(greet())


if __name__ == "__main__":
    main()
