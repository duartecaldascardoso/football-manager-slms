# Experiment with Small Language Models

Inspired on the article: https://arxiv.org/abs/2506.02153

Abstract:

    Large language models (LLMs) are often praised for exhibiting near-human performance 
    on a wide range of tasks and valued for their ability to hold a general
    conversation. The rise of agentic AI systems is, however, ushering in a mass of
    applications in which language models perform a small number of specialized tasks
    repetitively and with little variation.

    Here we lay out the position that small language models (SLMs) are sufficiently
    powerful, inherently more suitable, and necessarily more economical for many
    invocations in agentic systems, and are therefore the future of agentic AI. Our
    argumentation is grounded in the current level of capabilities exhibited by SLMs,
    the common architectures of agentic systems, and the economy of LM deployment.
    We further argue that in situations where general-purpose conversational abilities
    are essential, heterogeneous agentic systems (i.e., agents invoking multiple different
    models) are the natural choice. We discuss the potential barriers for the adoption
    of SLMs in agentic systems and outline a general LLM-to-SLM agent conversion
    algorithm.

    Our position, formulated as a value statement, highlights the significance of
    the operational and economic impact even a partial shift from LLMs to SLMs
    is to have on the AI agent industry. We aim to stimulate the discussion on
    the effective use of AI resources and hope to advance the efforts to lower
    the costs of AI of the present day. Calling for both contributions to and cri-
    tique of our position, we commit to publishing all such correspondence at

    research.nvidia.com/labs/lpr/slm-agent

The following experiment aims to practically understand some of the outlined advantages in the use of SLMs in Agentic use cases.

This experiment uses an agent with a set of tools and sub agents. The agents are used to create, manage and comment a football game.

