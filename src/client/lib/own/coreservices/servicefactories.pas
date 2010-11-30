unit servicefactories;
 {
   TServiceManager handles all services available to the GPU core.servicemanagers

   (c) by 2010 HB9TVM and the GPU Team
}

interface

uses
  servermanagers, dbtablemanagers, loggers,
  receivenodeservices;

type TServiceFactory = class(TObject)
   public
    constructor Create(servMan : TServerManager;
                       tableMan : TDbTableManager; proxy, port : String; logger : TLogger);
    destructor Destroy;

    function createReceiveNodeService() : TReceiveNodeServiceThread;

   private

     servMan_  : TServerManager;
     tableMan_ : TDbTableManager;
     logger_   : TLogger;
     proxy_,
     port_     : String;

end;

implementation

constructor TServiceFactory.Create(servMan : TServerManager;
                                   tableMan : TDbTableManager; proxy, port : String; logger : TLogger);
begin
 servMan_  := servMan;
 tableMan_ := tableMan;
 logger_   := logger;

 proxy_    := proxy;
 port_     := port;
end;

destructor TServiceFactory.Destroy;
begin
end;

function TServiceFactory.createReceiveNodeService() : TReceiveNodeServiceThread;
begin
 Result := TReceiveNodeServiceThread.Create(servMan_, proxy_, port_, tableMan_.getNodeTable(), logger_);
end;

end.
