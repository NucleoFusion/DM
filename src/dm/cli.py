import click
import pandas as pd
import numpy as np

@click.group()
def root():
    "Data Mining Project"

@click.command()
@click.option("--count", default=1, help="Number of greetings.")
@click.option("--surname", default="gupta", help="Surname.")
@click.argument("name")
@click.argument("midname")
def hello(count, surname,midname, name):
    arr = np.arange(count)
    df = pd.DataFrame({"No": arr + 1, "Greeting": [f"Hello, {name} {midname} {surname}"] * count})
    click.echo(df.to_string(index=False))

root.add_command(hello)

if __name__ == "__main__":
    root()
