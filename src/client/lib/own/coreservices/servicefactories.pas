unit servicefactories;
 {
   TServiceManager creates all services available to the GPU core.servicemanagers

   (c) by 2010 HB9TVM and the GPU Team
}

interface

uses
  servermanagers, dbtablemanagers, loggers,
  receivenodeservices, transmitnodeservices,
  receiveserverservices, coreconfigurations;

type TServiceFactory = class(TObject)
   public
    constructor Create(var servMan : TServerManager;
                       var tableMan : TDbTableManager; proxy, port : String; var logger : TLogger; var conf : TCoreConfiguration);
    destructor Destroy;

    function createReceiveNodeService()    : TReceiveNodeServiceThread;
    function createTransmitNodeService()   : TTransmitNodeServiceThread;
    function createReceiveServerService()  : TReceiveServerServiceThread;

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

function TServiceFactory.createReceiveNodeService() : TReceiveNodeServiceThread;
begin
 Result := TReceiveNodeServiceThread.Create(servMan_, proxy_, port_, tableMan_.getNodeTable(), logger_);
end;

function TServiceFactory.createTransmitNodeService() : TTransmitNodeServiceThread;
begin
 Result := TTransmitNodeServiceThread.Create(servMan_, proxy_, port_, logger_, conf_);
end;

function TServiceFactory.createReceiveServerService()  : TReceiveServerServiceThread;
begin
  Result := TReceiveServerServiceThread.Create(servMan_, proxy_, port_, tableMan_.getServerTable(), logger_, conf_);
end;




end.
