"""Optional provider integrations.

Each integration lives behind an extra (``pip install llm-metrics[openai]``) and
is imported from its own module, never from the package root, so installing the
SDK does not drag in every provider's client library.
"""
