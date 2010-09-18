{$DEFINE MSWINDOWS}
{$IFDEF VER170}
{$DEFINE D7}
{$ENDIF}
{$IFDEF VER150}
{$DEFINE D7}
{$ENDIF}
{
  In this unit, important structures of GPU are defined.
  TDllFunction is the signature for methods inside a DLL.
  Only TDllFunctions can be managed by the PluginManager.
  
  TStack is the internal structure used by plugins to communicate
  with the GPU core.
  
  TGPUIdentity definis physical details of a computer where
  GPU is running.
  
  TGPUCollectResult is used by the GPU component to collect
  results internally. However, frontends should collect
  results themselves. They cannot access this structure.
  
}
unit definitions;


interface

{$IFDEF MSWINDOWS}
uses Windows;

{$ENDIF}

const
  MAXSTACK = 128; {Maximum size of Stack in Virtual Machine}
  MAX_COLLECTING_IDS = 128; {Maximum number of Jobs we keep also average track}
  WRONG_PACKET_ID = 7777777; {if you get seven seven as result, the plugin returns false}
  INF      = 1.0 / 0.0;    {infinite to distinguish PChars from floats}
  APOSTROPHE = Chr(39);    {apostrophe is character to separate strings}
  QUOTE    = Chr(39);      {alias for apostrophe, '}

procedure GpuGetMem(var P: Pointer; Size: integer);
procedure GpuReallocMem(var P: Pointer; Size: integer);
procedure GpuFreeMem(var P: Pointer; Size: integer = 0);


type
  PStack = ^TStack;
  TSendCallBack = procedure(stk: PStack); stdcall;
  TSendStack = procedure; stdcall;



  TGpuGetMem = procedure(var P: Pointer; Size: integer); stdcall;
  TGpuReallocMem = procedure(var P: Pointer; Size: integer); stdcall;
  TGpuFreeMem = procedure(var P: Pointer); stdcall;


  TStack = record
    Stack: array [1..MAXSTACK] of double;
    StIdx: longint; {Index on Stack where Operations take place}
    {StIdx cannot be more than MAXSTACK}
    Progress: double;    {indicates plugin progress from 0 to 100}
    My: Pointer;
    {used to store data passed between update and function itself}

     {$IFDEF MSWINDOWS}
    hw:     HWND;
     {$ENDIF}
    Update: boolean; {function wants an update on the graphics window}

     {Stack for strings, PChar for compatibility with non
      Borland DLLs. If a value in stack is NAN, then StrStack
      is assigned to a PChar. If a value in StrStack is NULL,
      then a float is assigned in Stack.

      There is only
      one stack pointer for both Stack and StrStack.}
    PCharStack: array[1..MAXSTACK] of PChar;

     {for security reasons some jobs will be executed only
     locally}
    IsGlobal: boolean;

     {request to send back multiple results, works similarly
      to Update}
    MultipleResults: boolean;

    {callback for multiple results}
    SendCallback: TSendCallBack;
    SendStack:    TSendStack;

    //some vars to keep track of which is which and who is who:
    //     JobID: Integer;         //Specifies the jobID
    //     ThreadID: Integer;      //Thread handle of current running thread
    Thread: TObject;

    //for the dll's
    //     CustomTag: Integer;     //can be used by DLL if MultipleResults is true
    //     CustomPointer: Pointer; // "    "

    //The return stack of the plugin.
    //plugin needs to be called to cleanup this stack.  
    QCharStack: array[1..MAXSTACK] of PChar;

    //to share memory management with dll's
    //needed for exchanging PChars in a nice way,
    GpuGetMem:     TGpuGetMem;
    GpuReallocMem: TGpuReallocMem;
    GpuFreeMem:    TGpuFreeMem;
  end;

type
  TDllFunction = function(var stk: TStack): boolean; stdcall;


type   {here we collect results, computing average and so on}
  TGPUCollectResult = record
    TotalTime: TDateTime;
    FirstResult,
    LastResult: extended;
    N:      longint;
    Sum,                    {with sum and N we can compute average}
    Min,
    Max,
    Avg,
    SumStdDev,
    StdDev: extended;  {to compute stddev we need before average
                              we need to store all results, ahiahi,perhaps not implemented}
  end;

type
  TGPUIdentity = record
    NodeName,
    Team,
    Country,
    NodeId,
    IP,
    OS,
    Version:   string;
    Port : Longint; //TCP/IP Port
    AcceptIncoming : Boolean; // if a node is able to accept incoming connections
    SpeedMHz,
    RAM,
    Speedostones,
    Crawlostones,
    Terrastones,
    Computers: longint;
    PartLevel,
    PartLevelAggr: real; // hack for Netmapper algo, not elegant
    Terra:     integer;
    isSMP,               // the computer is a SMP computer with 2 CPUs
    isHT,                // the CPU has HyperThreading feature
    is64bit,              // the CPU is a 64 bit cpu
    isWineEmulator : boolean;
    isRunningAsScreensaver : boolean;
    MHzSecondCPU: longint;
    Uptime,
    TotalUptime : Double;
    Processor   : String;
    netmap_x,netmap_y, netmap_z : Double; // position inside netmapper 3D window

    Longitude,
    Latitude : Real;
  end;


const
  WRONG_GPU_COMMAND = 'UNKNOWN_COMMAND';

var
  MyGPUID: TGPUIdentity;


implementation

//maybe move to gpu_utils?
procedure GpuGetMem(var P: Pointer; Size: integer);
begin
  GetMem(P, Size);
end;

procedure GpuReallocMem(var P: Pointer; Size: integer);
begin
  ReallocMem(P, Size);
end;

procedure GpuFreeMem(var P: Pointer; Size: integer = 0);
begin
  if Size = 0 then
    FreeMem(P)
  else
    FreeMem(P, Size);
end;




end.
