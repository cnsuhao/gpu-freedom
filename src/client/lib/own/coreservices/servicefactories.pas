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
  restorestatusservices, receivejobstatusservices;

type TServiceFactory = class(TObject)
   public
    constructor Create(var workflowMan : TWorkflowManager; var servMan : TServerManager;
                       var tableMan : TDbTableManager; proxy, port : String; var logger : TLogger; var conf : TCoreConfiguration;
                       var coreModule : TCoreModule);
    destructor Destroy;

    function createReceiveClientService()  : TReceiveClientServiceThread;
    function createTransmitClientService() : TTransmitClientServiceThread;
    function createReceiveServerService()  : TReceiveServerServiceThread;
    function createReceiveParamService()   : TReceiveParamServiceThread;
    function createReceiveChannelService(channame, chantype : String) : TReceiveChannelServiceThread;
    function createTransmitChannelService(channame, chantype : String;
                                          content : AnsiString) : TTransmitChannelServiceThread;
    function createReceiveJobService() : TReceiveJobServiceThread;
    function createTransmitJobService() : TTransmitJobServiceThread;
    function createReceiveJobstatusService() : TReceiveJobstatusServiceThread;
    function createReceiveJobstatService() : TReceiveJobstatServiceThread;
    function createReceiveJobResultService() : TReceiveJobResultServiceThread;
    function createTransmitJobResultService() : TTransmitJobResultServiceThread;
    function createTransmitAckJobService() : TTransmitAckJobServiceThread;

    function createComputationService() : TComputationServiceThread;

    function createDownloadWUJobService() : TDownloadWUJobServiceThread;
    function createDownloadWUResultService() : TDownloadWUResultServiceThread;
    function createUploadWUJobService() : TUploadWUJobServiceThread;
    function createUploadWUResultService() : TUploadWUResultServiceThread;

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

function TServiceFactory.createReceiveClientService() : TReceiveClientServiceThread;
begin
 Result := TReceiveClientServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createTransmitClientService() : TTransmitClientServiceThread;
begin
 Result := TTransmitClientServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createReceiveServerService()  : TReceiveServerServiceThread;
begin
  Result := TReceiveServerServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createReceiveParamService() : TReceiveParamServiceThread;
begin
 Result := TReceiveParamServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createReceiveChannelService(channame, chantype : String) : TReceiveChannelServiceThread;
begin
  Result := TReceiveChannelServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_, channame, chantype);
end;

function TServiceFactory.createTransmitChannelService(channame, chantype : String;
                                                      content : AnsiString) : TTransmitChannelServiceThread;
begin
 Result := TTransmitChannelServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_, channame, chantype, content);
end;

function TServiceFactory.createReceiveJobService() : TReceiveJobServiceThread;
begin
 Result := TReceiveJobServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_, workflowman_);
end;

function TServiceFactory.createTransmitJobService() : TTransmitJobServiceThread;
begin
 Result := TTransmitJobServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_, workflowman_);
end;

function TServiceFactory.createReceiveJobstatusService() : TReceiveJobstatusServiceThread;
begin
 Result := TReceiveJobStatusServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_, workflowman_);
end;


function TServiceFactory.createReceiveJobstatService() : TReceiveJobstatServiceThread;
begin
 Result := TReceiveJobStatServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createReceiveJobResultService() : TReceiveJobResultServiceThread;
begin
 Result := TReceiveJobResultServiceThread.Create(servMan_,proxy_, port_, logger_, conf_, tableman_, workflowman_);
end;

function TServiceFactory.createTransmitJobResultService() : TTransmitJobResultServiceThread;
begin
 Result := TTransmitJobResultServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_, workflowman_);
end;

function TServiceFactory.createTransmitAckJobService() : TTransmitAckJobServiceThread;
begin
 Result := TTransmitAckJobServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableman_, workflowman_);
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


function TServiceFactory.createDownloadWUJobService() : TDownloadWUJobServiceThread;
begin
  Result := TDownloadWUJobServiceThread.Create(servMan_, tableman_, workflowman_, myConfId.proxy, myConfId.port, logger_);
end;

function TServiceFactory.createDownloadWUResultService() : TDownloadWUResultServiceThread;
begin
  Result := TDownloadWUResultServiceThread.Create(servMan_, tableman_, workflowman_, myConfId.proxy, myConfId.port, logger_);
end;

function TServiceFactory.createUploadWUJobService() : TUploadWUJobServiceThread;
begin
  Result := TUploadWUJobServiceThread.Create(servMan_,tableman_, workflowman_, myConfId.proxy, myConfId.port, logger_);
end;

function TServiceFactory.createUploadWUResultService() : TUploadWUResultServiceThread;
begin
  Result := TUploadWUResultServiceThread.Create(servMan_, tableman_, workflowman_, myConfId.proxy, myConfId.port, logger_);
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
