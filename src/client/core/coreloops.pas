unit coreloops;

interface

uses
  SysUtils,
  loggers, lockfiles, coreconfigurations,  identities,
  coremodules, servicefactories, servicemanagers,
  servermanagers, dbtablemanagers, coreservices,
  receiveparamservices, receiveserverservices,
  receiveclientservices, transmitclientservices,
  receivechannelservices, receivejobservices,
  receivejobstatservices,  receivejobresultservices,
  transmitjobservices, transmitjobresultservices,
  transmitackjobservices,
  jobdefinitiontables, jobresulttables, jobqueuetables,
  coreobjects, coremonitors;

const FRAC_SEC=1/24/3600;

type TCoreLoop = class(TObject)
  public
    constructor Create();
    destructor Destroy;

    procedure start;
    procedure tick;
    procedure stop;

    function getCoreMonitor : TCoreMonitor;

  private
    path_,
    logHeader_   : String;
    coremonitor_ : TCoreMonitor;

    tick_,
    days_        : Longint;

    function    launch(var thread : TCoreServiceThread; tname : String; var srv : TServerRecord) : Boolean;

    procedure   retrieveParams;
    procedure   retrieveServers;
    procedure   retrieveClients;
    procedure   retrieveChannels;
    procedure   retrieveJobs;
    procedure   retrieveJobStats;
    procedure   retrieveJobResults;

    procedure   transmitClient;
    procedure   transmitJob;
    procedure   transmitJobResult;
    procedure   transmitAck;
end;

implementation

constructor TCoreLoop.Create();
begin
  inherited Create;
  path_ := extractFilePath(ParamStr(0));
  logHeader_ := 'gpucore> ';

  coremonitor_ := TCoreMonitor.Create();
  loadCoreObjects('gpucore');
end;

destructor TCoreLoop.Destroy;
begin
 conf.saveCoreConfiguration();
 discardCoreObjects;
 coremonitor_.Free;
 inherited Destroy;
end;

procedure TCoreLoop.start;
begin
  coremonitor_.coreStarted;
  logger.logCR; logger.logCR;
  logger.logCR; logger.logCR;
  logger.log(LVL_INFO, logHeader_+'********************');
  logger.log(LVL_INFO, logHeader_+'* Core launched ...*');
  logger.log(LVL_INFO, logHeader_+'********************');
  // main loop
  tick_ := 1;
  days_ := 0;

  retrieveParams;
  Sleep(1000);
  //TODO: enable this!!
  //retrieveServers;
  Sleep(1000);
  retrieveClients;
  Sleep(1000);
  retrieveChannels;
  Sleep(1000);
  serviceman.clearFinishedThreads;
  transmitClient;
  Sleep(1000);
  retrieveJobs;
  Sleep(1000);
  retrieveJobResults;
  Sleep(1000);
  retrieveJobStats;

  { // u./sed to test Job transmission and Joresult transmission
  Sleep(1000);
  transmitJob;
  Sleep(1000);
  transmitJobResult;
  Sleep(1000);
  transmitAck;
  }
end;

procedure TCoreLoop.tick;
begin
      if (tick_ mod 60 = 0) then logger.log(LVL_DEBUG, logHeader_+'Running since '+FloatToStr(myGPUID.Uptime)+' days.');
      if (tick_ mod myConfID.receive_servers_each = 0) then begin retrieveParams; Sleep(1000); retrieveServers; end;
      if (tick_ mod myConfID.receive_nodes_each = 0) then retrieveClients;
      if (tick_ mod myConfID.transmit_node_each = 0) then transmitClient;
      if (tick_ mod myConfID.receive_channels_each = 0) then retrieveChannels;
      if lf_morefrequentupdates.exists  and (tick_ mod 20 = 0) then retrieveChannels;
      if (tick_ mod myConfId.receive_jobs_each = 0) then
          begin
            retrieveJobs;
            retrieveJobStats;
            retrieveJobResults;
          end;


      Inc(tick_);
      myGPUID.Uptime := myGPUID.Uptime+FRAC_SEC;

      if (tick_>=86400) then
         begin
            tick_ := 0;
            Inc(days_);
         end;
      serviceman.clearFinishedThreads;
end;

procedure TCoreLoop.stop;
begin
  logger.log(LVL_INFO, logHeader_+'Core was running for '+FloatToStr(myGPUID.uptime)+' days.');
  myGPUID.TotalUptime:=myGPUID.TotalUptime+myGPUID.Uptime;
  myGPUID.Uptime := 0;
  logger.log(LVL_INFO, logHeader_+'Total uptime is '+FloatToStr(myGPUID.TotalUptime)+'.');
  coremonitor_.coreStopped;
end;

function TCoreLoop.getCoreMonitor : TCoreMonitor;
begin
  Result := coremonitor_;
end;

function    TCoreLoop.launch(var thread : TCoreServiceThread; tname : String; var srv : TServerRecord) : Boolean;
var slot : Longint;
begin
   Result := true;
   logger.log(LVL_DEBUG, logHeader_+tname+' started...');
   slot := serviceman.launch(thread);
   if slot=-1 then
         begin
           Result := false;
           logger.log(LVL_SEVERE, logHeader_+tname+' failed, core too busy!');
         end;

   logger.log(LVL_DEBUG, logHeader_+tname+' over.');
end;


procedure TCoreLoop.retrieveParams;
var receiveparamthread  : TReceiveParamServiceThread;
    srv                 : TServerRecord;
begin
   serverman.getDefaultServer(srv);

   receiveparamthread  := servicefactory.createReceiveParamService(srv);
   if not launch( TCoreServiceThread(receiveparamthread), 'ReceiveParams', srv) then receiveparamthread.Free;
end;

procedure TCoreLoop.retrieveServers;
var receiveserverthread : TReceiveServerServiceThread;
    srv                 : TServerRecord;
begin
   serverman.getDefaultServer(srv);

   receiveserverthread := servicefactory.createReceiveServerService(srv);
   if not launch(TCoreServiceThread(receiveserverthread), 'ReceiveServers', srv) then receiveserverthread.Free;
end;

procedure TCoreLoop.retrieveClients;
var receiveclientthread  : TReceiveClientServiceThread;
    srv                  : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   receiveclientthread  := servicefactory.createReceiveClientService(srv);
   if not launch(TCoreServiceThread(receiveclientthread), 'ReceiveClients', srv) then receiveclientthread.Free;
end;

procedure TCoreLoop.retrieveChannels;
var receivechanthread     : TReceiveChannelServiceThread;
    srv                   : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   receivechanthread  := servicefactory.createReceiveChannelService(srv, {srv.chatchannel}'Altos', 'CHAT');
   if not launch(TCoreServiceThread(receivechanthread), 'ReceiveChannels', srv) then receivechanthread.Free;
end;

procedure TCoreLoop.retrieveJobs;
var receivejobthread     : TReceiveJobServiceThread;
    srv                  : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   receivejobthread  := servicefactory.createReceiveJobService(srv);
   if not launch(TCoreServiceThread(receivejobthread), 'ReceiveJobs', srv) then receivejobthread.Free;
end;

procedure TCoreLoop.retrieveJobStats;
var receivejobstatsthread     : TReceiveJobstatServiceThread;
    srv                       : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   receivejobstatsthread  := servicefactory.createReceiveJobstatService(srv);
   if not launch(TCoreServiceThread(receivejobstatsthread), 'ReceiveJobStats', srv) then receivejobstatsthread.Free;
end;


procedure TCoreLoop.retrieveJobResults;
var  receivejobresultthread : TReceiveJobresultServiceThread;
     srv                    : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   //TODO: we do not pass job id for the moment
   receivejobresultthread  := servicefactory.createReceiveJobresultService(srv, '');
   if not launch(TCoreServiceThread(receivejobresultthread), 'ReceiveJobResult', srv) then receivejobresultthread.Free;
end;


procedure TCoreLoop.transmitClient;
var transmitclientthread  : TTransmitClientServiceThread;
    srv                   : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   transmitclientthread  := servicefactory.createTransmitClientService(srv);
   if not launch(TCoreServiceThread(transmitclientthread), 'TransmitClient', srv) then transmitclientthread.Free;
end;

procedure TCoreLoop.transmitJob;
var transmitjobthread  : TTransmitJobServiceThread;
    srv                : TServerRecord;
    jobrow             : TDbJobDefinitionRow;
    trandetails        : TJobTransmissionDetails;
begin
   serverman.getDefaultServer(srv);

   // This was added for testing purposes
   jobrow.islocal:=false;
   jobrow.job:='13,12,add';
   jobrow.jobdefinitionid:='ajdflasdfjla';
   jobrow.jobtype:='GPU_Engine';
   jobrow.requireack:=true;
   jobrow.nodeid:=myGPUID.NodeId;
   jobrow.nodename:=myGPUID.NodeName;

   trandetails.nbrequests:=13;
   trandetails.tagwuresult:=false;
   trandetails.tagwujob:=false;
   trandetails.workunitjob:='';
   trandetails.workunitresult:='';

   transmitjobthread  := servicefactory.createTransmitJobService(srv, jobrow, trandetails);
   if not launch(TCoreServiceThread(transmitjobthread), 'TransmitJob', srv) then transmitjobthread.Free;
end;

procedure TCoreLoop.transmitJobResult;
var transmitjobresultthread  : TTransmitJobResultServiceThread;
    srv                      : TServerRecord;
    jobresultrow             : TDbJobResultRow;
begin
   serverman.getDefaultServer(srv);

    // This was added for testing purposes
   jobresultrow.nodename:= myGPUID.nodename;
   jobresultrow.nodeid  := myGPUID.nodeid;
   jobresultrow.jobdefinitionid:='ajdflasdfjla';
   jobresultrow.iserroneous:=false;
   jobresultrow.jobqueueid:='deec00759415e9201a21b0c197bb28b2';
   jobresultrow.jobresult:='25';
   jobresultrow.jobresultid:='asdfa';
   jobresultrow.errorid := 0;
   jobresultrow.errorarg := '';
   jobresultrow.errormsg := '';
   jobresultrow.workunitresult:='';
   transmitjobresultthread  := servicefactory.createTransmitJobResultService(srv, jobresultrow);
   if not launch(TCoreServiceThread(transmitjobresultthread), 'TransmitJobResult', srv) then transmitjobresultthread.Free;
end;

procedure TCoreLoop.transmitAck;
var transmitackthread  : TTransmitAckJobServiceThread;
    srv                : TServerRecord;
    jobqueuerow        : TDbJobQueueRow;
begin
  serverman.getDefaultServer(srv);

  // This was added for testing purposes
   jobqueuerow.jobqueueid:= 'deec00759415e9201a21b0c197bb28b2';
   jobqueuerow.jobdefinitionid     := 'ajdflasdfjla';
   jobqueuerow.status := JS_NEW;

  transmitackthread  := servicefactory.createTransmitAckJobService(srv, jobqueuerow);
  if not launch(TCoreServiceThread(transmitackthread), 'TransmitAckJob', srv) then transmitackthread.Free;
end;

end.
