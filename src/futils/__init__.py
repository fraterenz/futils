def parse_version(my_str: str) -> str:
    """Return the version from a rust binary, that is we assume something of
    this form: `'Dynamics 0.13.3\n'`
    """
    return "v" + my_str.replace("\n", "").split(" ")[-1]
