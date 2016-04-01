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
  receivegeoipservices,
  jobdefinitiontables,
  downloadservices, downloadservicemanagers,
  uploadservices, uploadservicemanagers,
  fasttransitionsfromnew, fasttransitionsfromcomputed, restorestatusservices,
  receivejobstatusservices,
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
    corenumber_  : Longint;

    tick_,
    days_        : Longint;
    started_     : Boolean;

    function    launch(var thread : TCoreServiceThread; tname : String) : Boolean;

    procedure   retrieveParams;
    procedure   retrieveServers;
    procedure   retrieveClients;
    procedure   retrieveChannels;
    procedure   retrieveJobStats;
    procedure   retrieveGeoIP;

    procedure   transmitClient;
    procedure   transmitJob;

    // client job processing worfklow
    procedure   retrieveJobs;
    procedure   createFastTransitionFromNewService;
    procedure   transmitAck;
    procedure   createDownloadWUJobService;
    procedure   createComputationService;
    procedure   createFastTransitionFromComputedService;
    procedure   createUploadWUResultService;
    procedure   transmitJobResult;


    // server job processing workflow
    procedure   createUploadWUJobService;
    procedure   createRetrieveStatusService;
    procedure   createDownloadWUResultService;
    procedure   retrieveJobResult;


    procedure   createRestoreStatusService;
end;

implementation

constructor TCoreLoop.Create();
begin
  started_ := False;
  inherited Create;
  path_ := extractFilePath(ParamStr(0));

  if Trim(ParamStr(1))<>'' then
    begin
      corenumber_ := StrToIntDef(ParamStr(1), -1);
      if corenumber_=-1 then corenumber_ := 1;
    end
  else
    corenumber_ := 1;

  coremonitor_ := TCoreMonitor.Create(corenumber_);
  logHeader_ := 'gpucore> ';

  loadCommonObjects('gpucore', 'GPU Core', corenumber_, true);
  loadCoreObjects();
end;

destructor TCoreLoop.Destroy;
begin
 coremonitor_.Free;
 discardCoreObjects;
 discardCommonObjects;
 inherited Destroy;
end;

procedure TCoreLoop.start;
begin
  coremonitor_.createLockFile;

  logger.log(LVL_INFO, logHeader_+'Bootstrapping core...');
  tick_ := 1;
  days_ := 0;

  createRestoreStatusService;
  retrieveParams;
  retrieveGeoIP;
  transmitClient;

  {
  //TODO: enable this!!
  //retrieveServers;
  }
  logger.log(LVL_INFO, logHeader_+'Core bootstrapped!');
  started_ := true;
end;

procedure TCoreLoop.tick;
var receiveServers,
    receiveJobs,
    transmitJobs    : Longint;
begin
    if not started_ then Exit;
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
      if (receiveJobs = 29) then retrieveJobStats;

      // ***************************************************
      // * Services for internal jobqueue workflow
      // ***************************************************
      // client processing workflow, see docs/dev/client-jobqueue-workflow.png
      // here, we process jobs received from a server
      if (tick_ mod 9 = 1)  then createFastTransitionFromNewService;
      if (tick_ mod 11 = 1) then createDownloadWUJobService;
      if (tick_ mod 13 = 1) then transmitAck;
      if (tick_ mod 17 = 1) then createComputationService;
      if (tick_ mod 21 = 1) then createFastTransitionFromComputedService;
      if (tick_ mod 23 = 1) then createUploadWUResultService;
      if (tick_ mod 27 = 1) then transmitJobResult;

      // server processing workflow, see docs/dev/server-jobqueue-workflow.png
      // here, we process jobs received from a server
      if (tick_ mod 29 = 1) then createUploadWUJobService;
      if (tick_ mod 31 = 1) then transmitJob;
      if (tick_ mod 33 = 1) then createRetrieveStatusService;
      if (tick_ mod 37 = 1) then createDownloadWUResultService;
      if (tick_ mod 43 = 1) then retrieveJobResult;

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
  conf.saveCoreConfiguration();
  logger.log(LVL_INFO, logHeader_+'Core configuration saved.');

  coremonitor_.removeLockFile;
end;

function TCoreLoop.getCoreMonitor : TCoreMonitor;
begin
  Result := coremonitor_;
end;

function    TCoreLoop.launch(var thread : TCoreServiceThread; tname : String) : Boolean;
var slot : Longint;
begin
   Result := true;

   //TODO: check if freeing inside launch does not create memory leak
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
begin
   receiveparamthread  := servicefactory.createReceiveParamService();
   launch( TCoreServiceThread(receiveparamthread), 'ReceiveParams');
end;

procedure TCoreLoop.retrieveServers;
var receiveserverthread : TReceiveServerServiceThread;
begin
   receiveserverthread := servicefactory.createReceiveServerService();
   launch(TCoreServiceThread(receiveserverthread), 'ReceiveServers');
end;

procedure TCoreLoop.retrieveClients;
var receiveclientthread  : TReceiveClientServiceThread;
begin
   receiveclientthread  := servicefactory.createReceiveClientService();
   launch(TCoreServiceThread(receiveclientthread), 'ReceiveClients');
end;

procedure TCoreLoop.retrieveChannels;
var receivechanthread     : TReceiveChannelServiceThread;
begin
   receivechanthread  := servicefactory.createReceiveChannelService({srv.chatchannel}'Altos', 'CHAT');
   launch(TCoreServiceThread(receivechanthread), 'ReceiveChannels');
end;

procedure TCoreLoop.retrieveJobs;
var receivejobthread     : TReceiveJobServiceThread;
begin
   receivejobthread  := servicefactory.createReceiveJobService();
   launch(TCoreServiceThread(receivejobthread), 'ReceiveJobs');
end;

procedure TCoreLoop.retrieveJobStats;
var receivejobstatsthread     : TReceiveJobstatServiceThread;
begin
   receivejobstatsthread  := servicefactory.createReceiveJobstatService();
   launch(TCoreServiceThread(receivejobstatsthread), 'ReceiveJobStats');
end;


procedure TCoreLoop.retrieveJobResult;
var  receivejobresultthread : TReceiveJobresultServiceThread;
begin
  receivejobresultthread  := servicefactory.createReceiveJobresultService();
  launch(TCoreServiceThread(receivejobresultthread), 'ReceiveJobResult');
end;

procedure TCoreLoop.retrieveGeoIP;
var  receivegeoipthread : TReceiveGeoIPServiceThread;
begin
  receivegeoipthread  := servicefactory.createReceiveGeoIPService();
  launch(TCoreServiceThread(receivegeoipthread), 'ReceiveGeoIP');
end;


procedure TCoreLoop.transmitClient;
var transmitclientthread  : TTransmitClientServiceThread;
begin
  transmitclientthread  := servicefactory.createTransmitClientService();
  launch(TCoreServiceThread(transmitclientthread), 'TransmitClient');
end;

procedure TCoreLoop.transmitJob;
var transmitjobthread  : TTransmitJobServiceThread;
begin
   transmitjobthread  := servicefactory.createTransmitJobService();
   launch(TCoreServiceThread(transmitjobthread), 'TransmitJob');
end;

procedure TCoreLoop.transmitJobResult;
var transmitjobresultthread  : TTransmitJobResultServiceThread;
    srv                      : TServerRecord;
begin
   transmitjobresultthread  := servicefactory.createTransmitJobResultService();
   launch(TCoreServiceThread(transmitjobresultthread), 'TransmitJobResult');
end;

procedure TCoreLoop.transmitAck;
var transmitackthread  : TTransmitAckJobServiceThread;
begin
  transmitackthread  := servicefactory.createTransmitAckJobService();
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

procedure TCoreLoop.createRestoreStatusService;
var thread : TRestoreStatusServiceThread;
begin
  thread := serviceFactory.createRestoreStatusService();
  launch(TCoreServiceThread(thread), 'RestoreStatusService');
end;


//************************************
//*  Client Job Processing Workflow
//************************************
procedure TCoreLoop.createDownloadWUJobService;
var downthread  : TDownloadWUJobServiceThread;
    slot        : Longint;
begin
  downThread := servicefactory.createDownloadWUJobService();
  slot := downserviceman.launch(TDownloadServiceThread(downthread));
  if slot=-1 then
      logger.log(LVL_SEVERE, logHeader_+'Download WU job service launch failed, core too busy!')
  else
      logger.log(LVL_DEBUG, logHeader_+'Download WU job service started...');
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


procedure  TCoreLoop.createUploadWUResultService;
var upthread  : TUploadWUResultServiceThread;
    slot      : Longint;
begin
  upThread := servicefactory.createUploadWUResultService();
  slot := upserviceman.launch(TUploadServiceThread(upthread));
  if slot=-1 then
      logger.log(LVL_SEVERE, logHeader_+'Upload WU result service launch failed, core too busy!')
  else
      logger.log(LVL_DEBUG, logHeader_+'Upload WU result service started...');
end;


//***********************************
//*  Server Job Processing Workflow
//***********************************
procedure  TCoreLoop.createUploadWUJobService;
var upthread  : TUploadWUJobServiceThread;
    slot      : Longint;
begin
  upThread := servicefactory.createUploadWUJobService();
  slot := upserviceman.launch(TUploadServiceThread(upthread));
  if slot=-1 then
      logger.log(LVL_SEVERE, logHeader_+'Upload WU Job service launch failed, core too busy!')
  else
      logger.log(LVL_DEBUG, logHeader_+'Upload WU job service started...');
end;


procedure TCoreLoop.createRetrieveStatusService;
var  receivejobstatusthread : TReceiveJobStatusServiceThread;
begin
  receivejobstatusthread  := servicefactory.createReceiveJobStatusService();
  launch(TCoreServiceThread(receivejobstatusthread), 'ReceiveJobStatus');
end;

procedure TCoreLoop.createDownloadWUResultService;
var downthread  : TDownloadWUResultServiceThread;
    slot        : Longint;
begin
   downThread := servicefactory.createDownloadWUResultService();
   slot := downserviceman.launch(TDownloadServiceThread(downthread));
   if slot=-1 then
           logger.log(LVL_SEVERE, logHeader_+'Download WU result service launch failed, core too busy!')
   else
      logger.log(LVL_DEBUG, logHeader_+'Download WU result service started...');
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
