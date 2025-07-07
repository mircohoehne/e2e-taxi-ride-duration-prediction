dev-setup:
	uv sync
	@read -p "Install pre-commit hooks? [y/N] " ans; \
	[ "$$ans" = "y" ] && uv run pre-commit install || echo "Skipping pre-commit hooks"
setup:
	uv sync --no-dev
data:
	echo "Not implemented yet"
