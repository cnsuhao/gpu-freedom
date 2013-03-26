unit coreloops;

interface

uses
  SysUtils,
  loggers, lockfiles, coreconfigurations,  identities,
  coremodules, servicefactories, servicemanagers, compservicemanagers,
  servermanagers, dbtablemanagers, coreservices,
  receiveparamservices, receiveserverservices,
  receiveclientservices, transmitclientservices,
  receivechannelservices, receivejobservices,
  receivejobstatservices,  receivejobresultservices,
  transmitjobservices, transmitjobresultservices,
  transmitackjobservices, computationservices,
  jobdefinitiontables,
  downloadservices, downloadservicemanagers,
  uploadservices, uploadservicemanagers,
  fasttransitionsfromnew, fasttransitionsfromcomputed,
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
    function waitingForShutdown : Boolean;

    procedure clearFinishedThreads;
    procedure printThreadManagersStatus;

  private
    path_,
    logHeader_   : String;
    coremonitor_ : TCoreMonitor;

    tick_,
    days_        : Longint;

    function    launch(var thread : TCoreServiceThread; tname : String) : Boolean;

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

    procedure   createFastTransitionFromNewService;
    procedure   createFastTransitionFromComputedService;
    procedure   createDownloadService;
    procedure   createComputationService;
    procedure   createUploadService;
end;

implementation

constructor TCoreLoop.Create();
begin
  inherited Create;
  path_ := extractFilePath(ParamStr(0));
  logHeader_ := 'gpucore> ';

  coremonitor_ := TCoreMonitor.Create();
  loadCoreObjects('gpucore', 'GPU Core');
end;

destructor TCoreLoop.Destroy;
begin
 discardCoreObjects;
 coremonitor_.Free;
 inherited Destroy;
end;

procedure TCoreLoop.start;
begin
  coremonitor_.coreStarted;
  // main loop
  tick_ := 1;
  days_ := 0;

  logger.log(LVL_INFO, logHeader_+'Bootstrapping core...');
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
  logger.log(LVL_INFO, logHeader_+'Core bootstrapped!');
end;

procedure TCoreLoop.tick;
var receiveServers,
    receiveJobs,
    transmitJobs    : Longint;
begin
      // ***************************************************
      // * Services for synchronization with server
      // ***************************************************
      receiveServers := tick_ mod myConfID.receive_servers_each;
      if (receiveServers = 3) then retrieveParams;
      //TODO: enable this
      //if (receiveServers = 4) then retrieveServers;

      if (tick_ mod myConfID.receive_nodes_each = 5) then retrieveClients;
      if (tick_ mod myConfID.transmit_node_each = 7) then transmitClient;
      if (tick_ mod myConfID.receive_channels_each = 11) then retrieveChannels;
      if lf_morefrequentupdates.exists  and (tick_ mod 13 = 0) then retrieveChannels;

      receiveJobs := tick_ mod myConfId.receive_jobs_each;
      if (receiveJobs = 0)  then retrieveJobs;
      if (receiveJobs = 13) then retrieveJobResults;
      if (receiveJobs = 29) then retrieveJobStats;

      // TODO: how is handling of transmitting jobs regulated?
      // isn't this a task for the GUI?
      // transmitJobs := tick_ mod myConfId.transmit_jobs_each;
      // if (transmitJobs = 0) then transmitJob;


      // ***************************************************
      // * Services for internal jobqueue workflow
      // ***************************************************
      // TODO: decide here how the coreloop functionality has to tick
      if (tick_ mod 9 = 1)  then createFastTransitionFromNewService;
      if (tick_ mod 11 = 1) then createDownloadService;
      if (tick_ mod 13 = 1) then transmitAck;
      if (tick_ mod 17 = 1) then createComputationService;
      if (tick_ mod 21 = 1) then createFastTransitionFromComputedService;
      if (tick_ mod 23 = 1) then createUploadService;
      if (tick_ mod 27 = 1) then transmitJobResult;

      // ***************************************************
      // * TCoreLoop clock
      // ***************************************************
      Inc(tick_);
      myGPUID.Uptime := myGPUID.Uptime+FRAC_SEC;
      if (tick_>=86400) then
         begin
            tick_ := 0;
            Inc(days_);
         end;

      if (tick_ mod 60 = 0) then logger.log(LVL_DEBUG, logHeader_+'Running since '+FloatToStr(myGPUID.Uptime)+' days.');
      if (tick_ mod 120 = 0) then printThreadManagersStatus;

      // ***************************************************
      // * Updating internal status of service managers
      // ***************************************************
      clearFinishedThreads;
end;

function TCoreLoop.waitingForShutdown : Boolean;
begin
   serviceman.clearFinishedThreads;
   Result := not serviceman.isIdle();
end;

procedure TCoreLoop.stop;
begin
  logger.log(LVL_INFO, logHeader_+'Core was running for '+FloatToStr(myGPUID.uptime)+' days.');
  myGPUID.TotalUptime:=myGPUID.TotalUptime+myGPUID.Uptime;
  myGPUID.Uptime := 0;
  logger.log(LVL_INFO, logHeader_+'Total uptime is '+FloatToStr(myGPUID.TotalUptime)+'.');
  coremonitor_.coreStopped;
  conf.saveCoreConfiguration();
  logger.log(LVL_INFO, logHeader_+'Core configuration saved.');
end;

function TCoreLoop.getCoreMonitor : TCoreMonitor;
begin
  Result := coremonitor_;
end;

function    TCoreLoop.launch(var thread : TCoreServiceThread; tname : String) : Boolean;
var slot : Longint;
begin
   Result := true;
   slot := serviceman.launch(thread, tname);
   if slot=-1 then
         begin
           Result := false;
           logger.log(LVL_SEVERE, logHeader_+tname+' failed, core too busy!');
         end
   else
        logger.log(LVL_DEBUG, logHeader_+tname+' started...');
end;


procedure TCoreLoop.retrieveParams;
var receiveparamthread  : TReceiveParamServiceThread;
    srv                 : TServerRecord;
begin
   serverman.getDefaultServer(srv);

   receiveparamthread  := servicefactory.createReceiveParamService(srv);
   launch( TCoreServiceThread(receiveparamthread), 'ReceiveParams');
end;

procedure TCoreLoop.retrieveServers;
var receiveserverthread : TReceiveServerServiceThread;
    srv                 : TServerRecord;
begin
   serverman.getDefaultServer(srv);

   receiveserverthread := servicefactory.createReceiveServerService(srv);
   launch(TCoreServiceThread(receiveserverthread), 'ReceiveServers');
end;

procedure TCoreLoop.retrieveClients;
var receiveclientthread  : TReceiveClientServiceThread;
    srv                  : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   receiveclientthread  := servicefactory.createReceiveClientService(srv);
   launch(TCoreServiceThread(receiveclientthread), 'ReceiveClients');
end;

procedure TCoreLoop.retrieveChannels;
var receivechanthread     : TReceiveChannelServiceThread;
    srv                   : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   receivechanthread  := servicefactory.createReceiveChannelService(srv, {srv.chatchannel}'Altos', 'CHAT');
   launch(TCoreServiceThread(receivechanthread), 'ReceiveChannels');
end;

procedure TCoreLoop.retrieveJobs;
var receivejobthread     : TReceiveJobServiceThread;
    srv                  : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   receivejobthread  := servicefactory.createReceiveJobService(srv);
   launch(TCoreServiceThread(receivejobthread), 'ReceiveJobs');
end;

procedure TCoreLoop.retrieveJobStats;
var receivejobstatsthread     : TReceiveJobstatServiceThread;
    srv                       : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   receivejobstatsthread  := servicefactory.createReceiveJobstatService(srv);
   launch(TCoreServiceThread(receivejobstatsthread), 'ReceiveJobStats');
end;


procedure TCoreLoop.retrieveJobResults;
var  receivejobresultthread : TReceiveJobresultServiceThread;
     srv                    : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   //TODO: we do not pass job id for the moment
   receivejobresultthread  := servicefactory.createReceiveJobresultService(srv, '');
  launch(TCoreServiceThread(receivejobresultthread), 'ReceiveJobResult');
end;


procedure TCoreLoop.transmitClient;
var transmitclientthread  : TTransmitClientServiceThread;
    srv                   : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   transmitclientthread  := servicefactory.createTransmitClientService(srv);
  launch(TCoreServiceThread(transmitclientthread), 'TransmitClient');
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
   launch(TCoreServiceThread(transmitjobthread), 'TransmitJob');
end;

procedure TCoreLoop.transmitJobResult;
var transmitjobresultthread  : TTransmitJobResultServiceThread;
    srv                      : TServerRecord;
begin
   serverman.getDefaultServer(srv);

   transmitjobresultthread  := servicefactory.createTransmitJobResultService(srv);
   launch(TCoreServiceThread(transmitjobresultthread), 'TransmitJobResult');
end;

procedure TCoreLoop.transmitAck;
var transmitackthread  : TTransmitAckJobServiceThread;
    srv                : TServerRecord;
begin
  serverman.getDefaultServer(srv);
  transmitackthread  := servicefactory.createTransmitAckJobService(srv);
  launch(TCoreServiceThread(transmitackthread), 'TransmitAckJob');
end;

procedure TCoreLoop.createFastTransitionFromNewService;
var thread : TFastTransitionFromNewServiceThread;
begin
  thread := serviceFactory.createFastTransitionFromNewService();
  launch(TCoreServiceThread(thread), 'FastTransitionFromNew');
end;

procedure TCoreLoop.createFastTransitionFromComputedService;
var thread : TFastTransitionFromComputedServiceThread;
begin
  thread := serviceFactory.createFastTransitionFromComputedService();
  launch(TCoreServiceThread(thread), 'FastTransitionFromComputed');
end;

procedure TCoreLoop.createDownloadService;
var downthread  : TDownloadServiceThread;
    srv         : TServerRecord;
    slot        : Longint;
begin
  serverman.getDefaultServer(srv);
  downThread := servicefactory.createDownloadService(srv);

   slot := downserviceman.launch(downthread);
   if slot=-1 then
           logger.log(LVL_SEVERE, logHeader_+'Download service launch failed, core too busy!')
   else
      logger.log(LVL_DEBUG, logHeader_+'Download service started...');
end;

procedure TCoreLoop.createComputationService;
var compthread : TComputationServiceThread;
    slot       : Longint;
begin
   compthread := serviceFactory.createComputationService();

   slot := compserviceman.launch(compthread);
   if slot=-1 then
           logger.log(LVL_SEVERE, logHeader_+'Computation service launch failed, core too busy!')
   else
      logger.log(LVL_DEBUG, logHeader_+'Computation service started...');
end;

procedure  TCoreLoop.createUploadService;
var upthread  : TUploadServiceThread;
    srv       : TServerRecord;
    slot      : Longint;
begin
  serverman.getDefaultServer(srv);
  upThread := servicefactory.createUploadService(srv);
  slot := upserviceman.launch(upthread);
  if slot=-1 then
      logger.log(LVL_SEVERE, logHeader_+'Upload service launch failed, core too busy!')
  else
      logger.log(LVL_DEBUG, logHeader_+'Upload service started...');
end;

procedure TCoreLoop.clearFinishedThreads;
begin
  serviceman.clearFinishedThreads;
  compserviceman.clearFinishedThreads;
  downserviceman.clearFinishedThreads;
  upserviceman.clearFinishedThreads;
end;

procedure TCoreLoop.printThreadManagersStatus;
begin
  serviceman.printThreadStatus('Service Manager', logger);
  compserviceman.printThreadStatus('Computation Manager', logger);
  downserviceman.printThreadStatus('Download Manager', logger);
  upserviceman.printThreadStatus('Upload Manager', logger);
end;

end.
