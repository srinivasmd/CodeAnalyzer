from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="code-review-summarizer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An intelligent code review assistant that analyzes pull requests like a senior developer",
    long_description="""
    Code Review Summarizer is an AI-powered tool that helps streamline the code review process by:
    - Analyzing pull requests for security vulnerabilities and architectural changes
    - Distinguishing between cosmetic and substantive changes
    - Providing focused review suggestions with deep-dive links
    - Integrating multiple static analysis tools
    - Learning from historical code review patterns
    """,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "code-review-summarizer=code_review_summarizer.main:main"
        ],
    },
)