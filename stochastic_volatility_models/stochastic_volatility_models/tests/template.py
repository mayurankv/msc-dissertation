# Imports
from typer import Typer
from rich import print

# Initialise Decorators
app = Typer()


# Main Function
@app.command()
def main():
	# TODO (@mayurankv): Implement
	print(":exclamation: [red]To be implemented[/red]")


# Guarded Script
if __name__ == "__main__":
	app()
