"""Minimal demo app for the dsh-superagent sandbox."""


def greet(name="world"):
    return f"Hello, {name}!"


def main():
    print(greet())


if __name__ == "__main__":
    main()
