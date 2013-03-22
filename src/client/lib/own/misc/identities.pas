unit identities;
{
 In this unit:
 
TGPUIdentity defines physical details of a computer where
  GPU is running.
  
  TUserIdentity defines details of a user. A user can run more
  than one computer for GPU.

}
interface

type
  TGPUIdentity = record
    NodeName,
    NodeId,
    Team,
    Country,
    Region,
    Street,
    City,
    Zip,
    description,
    IP,
    localIP,
    OS,
    Port : String;
    AcceptIncoming : Boolean; // if a node is able to accept incoming connections
    MHz,
    RAM,
    GigaFlops,
    bits         : Longint;
    isScreensaver : Boolean;
    nbCPUs: Longint;
    Version,
    Uptime,
    TotalUptime : Double;
    CPUType     : String;

    Longitude,
    Latitude : Extended;
  end;
  
type 
  TUserIdentity = record
     userid,
     username,
     password,
     email,
     realname,
     homepage_url : String;
  end;

type TConfIdentity = record
    max_computations,
    max_services,
    max_downloads,
    max_uploads : Longint;

    run_only_when_idle : boolean;

    proxy,
    port,
    default_superserver_url,
    default_server_name      : String;

    receive_servers_each,
    receive_nodes_each,
    transmit_node_each,
    receive_jobs_each,
    transmit_jobs_each,
    receive_channels_each,
    transmit_channels_each,
    receive_chat_each,
    purge_server_after_failures : Longint;

    loglevel : Longint;
end;

type
  TThreadManagerStatus = record
     threads,
     maxthreads : Longint;

     isIdle,
     hasResources : Boolean;
  end;


var
  MyGPUID         : TGPUIdentity;
  MyUserID        : TUserIdentity;
  MyConfID        : TConfIdentity;

  tmCompStatus    : TThreadManagerStatus;
  tmDownStatus    : TThreadManagerStatus;
  tmUploadStatus  : TThreadManagerStatus;
  tmServiceStatus : TThreadManagerStatus;

implementation



end.
