









> Use this file to discover all available pages before exploring further.
Before your voice AI bot can start processing audio and generating responses, you need to establish a connection between the user and your bot. This process is called - it’s how users and bots find each other and establish a communication channel for real-time audio exchange.
  * : A FastAPI server that handles incoming connection requests and manages session setup
  * : Your voice AI application running as a separate server-side service
  * : The user-facing app (web browser, mobile app, etc.)

The runner acts as the coordinator, setting up the necessary resources and starting bot instances, while the Pipecat bot handles the actual voice AI processing. For most development and many production use cases, Pipecat provides a that handles all the session initialization complexity for you. Instead of building FastAPI servers and managing WebRTC connections yourself, you focus on your bot logic while the runner handles the infrastructure. Your bot needs a single entry point function that the runner will call:

```





















```


```









```

where specifies the transport type (e.g., , , ) and is the optional proxy domain for telephony.


Learn more about building with the development runner in the .
While the development runner handles the complexity, understanding the three connection patterns helps you choose the right approach and debug issues:
  1. When you open the page and connect, browser creates a WebRTC offer


User visits the client application and clicks to start a session
Runner calls Daily’s API to create a room and tokens using 
Both user’s browser and your bot join the same Daily room
Room-based WebRTC can also be used for SIP or PSTN connections, which require different connection patterns. Refer to the for details.

How and when your bot begins talking depends on the connection type: These connections are ready immediately, so you can start talking right after connection: For client/server applications using room-based WebRTC, a handshake ensures both sides are ready and the client won’t miss the opening message:
For client/server room-based connections, waiting for is crucial - starting too early can cause the client to miss part of the initial message.

The development runner works for most cases, but sometimes you need custom behavior - specific authentication, custom endpoints, or integration with existing systems. For these cases, you can create your own FastAPI runner. The development runner source code () provides excellent examples for:

```












```


```












```

Refer to the development runner source code to understand these patterns before building custom runners. It handles many edge cases and provides battle-tested implementations.
  * to choose the right approach for your use case
  * - immediate start vs. handshake patterns matter
  * - one bot instance per session is the recommended pattern

Now that you understand session initialization, let’s explore the different transport options and how to configure them for your specific needs.
Learn how Pipecat’s pipeline architecture orchestrates frame processing for voice AI applications
