unit servicefactories;
 {
   TServiceFactory creates all services available to the GPU core.

   (c) by 2002-2013 HB9TVM and the GPU Team
}

interface

uses
  servermanagers, dbtablemanagers, loggers,
  receiveclientservices, transmitclientservices,
  receiveserverservices, receiveparamservices,
  receivechannelservices, transmitchannelservices,
  receivejobservices, transmitjobservices,
  receivejobresultservices, transmitjobresultservices,
  receivejobstatservices, transmitackjobservices, workflowmanagers,
  coreconfigurations, jobdefinitiontables, jobresulttables,
  computationservices, coremodules, pluginmanagers, frontendmanagers,
  methodcontrollers, resultcollectors,
  downloadservices, uploadservices, identities,
  fasttransitionsfromnew, fasttransitionsfromcomputed,
  restorestatusservices;

type TServiceFactory = class(TObject)
   public
    constructor Create(var workflowMan : TWorkflowManager; var servMan : TServerManager;
                       var tableMan : TDbTableManager; proxy, port : String; var logger : TLogger; var conf : TCoreConfiguration;
                       var coreModule : TCoreModule);
    destructor Destroy;

    function createReceiveClientService(var srv : TServerRecord)  : TReceiveClientServiceThread;
    function createTransmitClientService(var srv : TServerRecord) : TTransmitClientServiceThread;
    function createReceiveServerService(var srv : TServerRecord)  : TReceiveServerServiceThread;
    function createReceiveParamService(var srv : TServerRecord)   : TReceiveParamServiceThread;
    function createReceiveChannelService(var srv : TServerRecord;
                                         channame, chantype : String) : TReceiveChannelServiceThread;
    function createTransmitChannelService(var srv : TServerRecord;
                                          channame, chantype : String;
                                          content : AnsiString) : TTransmitChannelServiceThread;
    function createReceiveJobService(var srv : TServerRecord) : TReceiveJobServiceThread;
    function createTransmitJobService(var srv : TServerRecord; var jobrow : TDbJobDefinitionRow; var trandetails : TJobTransmissionDetails) : TTransmitJobServiceThread;
    function createReceiveJobstatService(var srv : TServerRecord) : TReceiveJobstatServiceThread;
    function createReceiveJobResultService(var srv : TServerRecord; jobid : String) : TReceiveJobResultServiceThread;
    function createTransmitJobResultService(var srv : TServerRecord) : TTransmitJobResultServiceThread;
    function createTransmitAckJobService(var srv : TServerRecord) : TTransmitAckJobServiceThread;

    function createComputationService() : TComputationServiceThread;

    function createDownloadWUJobService(var srv : TServerRecord) : TDownloadWUJobServiceThread;
    function createDownloadWUResultService(var srv : TServerRecord) : TDownloadWUResultServiceThread;
    function createUploadWUJobService(var srv : TServerRecord) : TUploadWUJobServiceThread;
    function createUploadWUResultService(var srv : TServerRecord) : TUploadWUResultServiceThread;

    function createFastTransitionFromNewService() : TFastTransitionFromNewServiceThread;
    function createFastTransitionFromComputedService() : TFastTransitionFromComputedServiceThread;

    function createRestoreStatusService() : TRestoreStatusServiceThread;

   private

     servMan_      : TServerManager;
     tableMan_     : TDbTableManager;
     workflowMan_  : TWorkflowManager;
     logger_       : TLogger;
     proxy_,
     port_         : String;
     conf_         : TCoreConfiguration;
     core_         : TCoreModule;


end;

implementation

constructor TServiceFactory.Create(var workflowMan : TWorkflowManager; var servMan : TServerManager;
                                   var tableMan : TDbTableManager; proxy, port : String; var logger : TLogger; var conf : TCoreConfiguration;
                                   var coreModule : TCoreModule);
begin
 servMan_  := servMan;
 tableMan_ := tableMan;
 logger_   := logger;
 workflowMan_ := workflowMan;
 core_     := coreModule;

 proxy_    := proxy;
 port_     := port;
 conf_     := conf;
end;

destructor TServiceFactory.Destroy;
begin
 inherited Destroy;
end;

function TServiceFactory.createReceiveClientService(var srv : TServerRecord) : TReceiveClientServiceThread;
begin
 Result := TReceiveClientServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createTransmitClientService(var srv : TServerRecord) : TTransmitClientServiceThread;
begin
 Result := TTransmitClientServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createReceiveServerService(var srv : TServerRecord)  : TReceiveServerServiceThread;
begin
  Result := TReceiveServerServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createReceiveParamService(var srv : TServerRecord) : TReceiveParamServiceThread;
begin
 Result := TReceiveParamServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createReceiveChannelService(var srv : TServerRecord;
                                     channame, chantype : String) : TReceiveChannelServiceThread;
begin
  Result := TReceiveChannelServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_, channame, chantype);
end;

function TServiceFactory.createTransmitChannelService(var srv : TServerRecord;
                                                      channame, chantype : String;
                                                      content : AnsiString) : TTransmitChannelServiceThread;
begin
 Result := TTransmitChannelServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_, channame, chantype, content);
end;

function TServiceFactory.createReceiveJobService(var srv : TServerRecord) : TReceiveJobServiceThread;
begin
 Result := TReceiveJobServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_, workflowman_);
end;

function TServiceFactory.createTransmitJobService(var srv : TServerRecord; var jobrow : TDbJobDefinitionRow; var trandetails : TJobTransmissionDetails) : TTransmitJobServiceThread;
begin
 Result := TTransmitJobServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_, jobrow, trandetails);
end;

function TServiceFactory.createReceiveJobstatService(var srv : TServerRecord) : TReceiveJobstatServiceThread;
begin
 Result := TReceiveJobStatServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createReceiveJobResultService(var srv : TServerRecord; jobid : String) : TReceiveJobResultServiceThread;
begin
 Result := TReceiveJobResultServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_, jobid);
end;

function TServiceFactory.createTransmitJobResultService(var srv : TServerRecord) : TTransmitJobResultServiceThread;
begin
 Result := TTransmitJobResultServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_, workflowman_);
end;

function TServiceFactory.createTransmitAckJobService(var srv : TServerRecord) : TTransmitAckJobServiceThread;
begin
 Result := TTransmitAckJobServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_, workflowman_);
end;

function TServiceFactory.createComputationService() : TComputationServiceThread;
var plugman : TPluginManager;
    meth    : TMethodController;
    res     : TResultCollector;
    front   : TFrontendManager;
begin
 plugman := core_.getPluginManager();
 meth    := core_.getMethController();
 res     := core_.getResultCollector();
 front   := core_.getFrontendManager();
 Result := TComputationServiceThread.Create(plugman, meth, res, front, workflowman_, tableman_, logger_);
end;


function TServiceFactory.createDownloadWUJobService(var srv : TServerRecord) : TDownloadWUJobServiceThread;
begin
  Result := TDownloadWUJobServiceThread.Create(srv, tableman_, workflowman_, myConfId.proxy, myConfId.port, logger_);
end;

function TServiceFactory.createDownloadWUResultService(var srv : TServerRecord) : TDownloadWUResultServiceThread;
begin
  Result := TDownloadWUResultServiceThread.Create(srv, tableman_, workflowman_, myConfId.proxy, myConfId.port, logger_);
end;

function TServiceFactory.createUploadWUJobService(var srv : TServerRecord) : TUploadWUJobServiceThread;
begin
  Result := TUploadWUJobServiceThread.Create(srv, tableman_, workflowman_, myConfId.proxy, myConfId.port, logger_);
end;

function TServiceFactory.createUploadWUResultService(var srv : TServerRecord) : TUploadWUResultServiceThread;
begin
  Result := TUploadWUResultServiceThread.Create(srv, tableman_, workflowman_, myConfId.proxy, myConfId.port, logger_);
end;

function TServiceFactory.createFastTransitionFromNewService() : TFastTransitionFromNewServiceThread;
begin
  Result := TFastTransitionFromNewServiceThread.Create(logger_, '[TFastTransitionFromNew]> ', conf_, tableman_, workflowman_);
end;

function TServiceFactory.createRestoreStatusService() : TRestoreStatusServiceThread;
begin
   Result := TRestoreStatusServiceThread.Create(logger_, '[TRestoreStatusService]> ', conf_, tableman_, workflowman_);
end;

function TServiceFactory.createFastTransitionFromComputedService() : TFastTransitionFromComputedServiceThread;
begin
  Result := TFastTransitionFromComputedServiceThread.Create(logger_, '[TFastTransitionFromComputed]> ', conf_, tableman_, workflowman_);
end;

end.
