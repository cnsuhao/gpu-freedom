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
    Team,
    Country,
    Region,
    Street,
    City,
    Zip,
    NodeId,
    IP,
    localIP,
    OS,
    Version:   string;
    Port : Longint; //TCP/IP Port
    AcceptIncoming : Boolean; // if a node is able to accept incoming connections
    MHz,
    RAM,
    GigaFlops : Longint;
    isSMP,               // the computer is a SMP computer with more than 1 CPU case box
    isHT,                // the CPU has HyperThreading feature
    is64bit,              // the CPU is a 64 bit cpu
    isWineEmulator : Boolean;
    isRunningAsScreensaver : Boolean;
    nbCPUs: Longint;
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
    max_downloads : Longint;

    default_superserver_url : String;

    receive_servers_each,
    receive_nodes_each,
    transmit_node_each,
    receive_jobs_each,
    transmit_jobs_each,
    receive_channels_each,
    transmit_channels_each,
    receive_chat_each,
    purge_server_after_failures : Longint;
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
  TMCompStatus    : TThreadManagerStatus;
  TMDownStatus    : TThreadManagerStatus;
  TMServiceStatus : TThreadManagerStatus;

implementation



end.
