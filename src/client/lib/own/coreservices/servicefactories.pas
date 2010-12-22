unit servicefactories;
 {
   TServiceFactory creates all services available to the GPU core.servicemanagers

   (c) by 2010 HB9TVM and the GPU Team
}

interface

uses
  servermanagers, dbtablemanagers, loggers,
  receiveclientservices, transmitclientservices,
  receiveserverservices, receiveparamservices,
  receivechannelservices, coreconfigurations;

type TServiceFactory = class(TObject)
   public
    constructor Create(var servMan : TServerManager;
                       var tableMan : TDbTableManager; proxy, port : String; var logger : TLogger; var conf : TCoreConfiguration);
    destructor Destroy;

    function createReceiveClientService()  : TReceiveClientServiceThread;
    function createTransmitClientService() : TTransmitClientServiceThread;
    function createReceiveServerService()  : TReceiveServerServiceThread;
    function createReceiveParamService()   : TReceiveParamServiceThread;
    function createReceiveChannelService(var srv : TServerRecord;
                                         channame, chantype : String) : TReceiveChannelServiceThread;

   private

     servMan_  : TServerManager;
     tableMan_ : TDbTableManager;
     logger_   : TLogger;
     proxy_,
     port_     : String;
     conf_     : TCoreConfiguration;

end;

implementation

constructor TServiceFactory.Create(var servMan : TServerManager;
                                   var tableMan : TDbTableManager; proxy, port : String; var logger : TLogger; var conf : TCoreConfiguration);
begin
 servMan_  := servMan;
 tableMan_ := tableMan;
 logger_   := logger;

 proxy_    := proxy;
 port_     := port;
 conf_     := conf;
end;

destructor TServiceFactory.Destroy;
begin
end;

function TServiceFactory.createReceiveClientService() : TReceiveClientServiceThread;
begin
 Result := TReceiveClientServiceThread.Create(servMan_, proxy_, port_, tableMan_.getClientTable(), logger_, conf_);
end;

function TServiceFactory.createTransmitClientService() : TTransmitClientServiceThread;
begin
 Result := TTransmitClientServiceThread.Create(servMan_, proxy_, port_, logger_, conf_, tableMan_.getClientTable());
end;

function TServiceFactory.createReceiveServerService()  : TReceiveServerServiceThread;
begin
  Result := TReceiveServerServiceThread.Create(servMan_, proxy_, port_, tableMan_.getServerTable(), logger_, conf_);
end;

function TServiceFactory.createReceiveParamService() : TReceiveParamServiceThread;
begin
 Result := TReceiveParamServiceThread.Create(servMan_, proxy_, port_, logger_, conf_);
end;

function TServiceFactory.createReceiveChannelService(var srv : TServerRecord;
                                     channame, chantype : String) : TReceiveChannelServiceThread;
begin
  Result := TReceiveChannelServiceThread.Create(servMan_, proxy_, port_, tableMan_, logger_, srv, channame, chantype);
end;


end.
