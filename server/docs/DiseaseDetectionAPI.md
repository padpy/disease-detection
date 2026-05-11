# Disease Detection API
This documents the disease dection API RESTful sequence diagram. This defines the process for submitting a disease deteciton plant, and how a client should retrieve data. This sequence diagram only contains that Client and API layer, as the architecture of the the Application layer and Data layers have not been decided.

The plant submission should contain an image of the crop to perform disease detection on. The server will respond with an a GUID that will serve as the plant ID for retrieving the data. The client then can request the plant status, and retrieve the plant data and metadata when the plant is completed. Data that can not be transmitted over with the initial plant data request, like images, will be requirested individually by the client.

```mermaid
sequenceDiagram
participant c as Client
participant a as API
note over c, a: Disease Detection plant submission
c->>a: Crop Image (PUT)
a->>c: plant ID (Response)

note over c, a: plant status check. <br> Repeat till plant completed...
c->>a: plant Status (GET)
a->>c: 

note over c, a: Get plant data/metadata
c->>a: plant data (GET)
a->>c: 

note over c, a: Get specific data that cannot be <br> submitted with data (Images, ....)
c->>a: plant entry specific data (GET)
a->>c: 

```
