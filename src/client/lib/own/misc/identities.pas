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
    NodeId,
    IP,
    OS,
    Version:   string;
    Port : Longint; //TCP/IP Port
    AcceptIncoming : Boolean; // if a node is able to accept incoming connections
    MHz,
    RAM,
    GigaFlops : Longint
    isSMP,               // the computer is a SMP computer with more than 1 CPUs
    isHT,                // the CPU has HyperThreading feature
    is64bit,              // the CPU is a 64 bit cpu
    isWineEmulator : Boolean;
    isRunningAsScreensaver : Boolean;
    nbCPUs: Longint;
    Uptime,
    TotalUptime : Double;
    Processor   : String;

    Longitude,
    Latitude : Extended;
  end;
  
type 
  TUserIdentity = record
     userid,
     username,
     password,
     homepage_url : String;
  end;



var
  MyGPUID: TGPUIdentity;
  MyUserID : TUserIdentity;

implementation



end.