unit servicemanagers;
 {
   TServiceManager handles all services available to the GPU core.servicemanagers

   (c) by 2010 HB9TVM and the GPU Team
}

interface

uses
  servermanagers, dbtablemanagers, loggers,
  receivenodeservices;

type TServiceManager = class(TObject)
   public
    constructor Create(servMan : TServerManager;
                       tableMan : TDbTableManager; proxy, port : String; logger : TLogger);
    destructor Destroy;

    function createReceivenodeservice() : TReceiveNodeServiceThread;

   private

     servMan_  : TServerManager;
     tableMan_ : TDbTableManager;
     logger_   : TLogger;
     proxy_,
     port_     : String;

end;

implementation

constructor TServiceManager.Create(servMan : TServerManager;
                                   tableMan : TDbTableManager; proxy, port : String; logger : TLogger);
begin
 servMan_  := servMan;
 tableMan_ := tableMan_;
 logger_   := logger_;

 proxy_    := proxy;
 port_     := port;
end;

destructor TServiceManager.Destroy;
begin
end;

function TServiceManager.createReceivenodeservice() : TReceiveNodeServiceThread;
begin
 Result := TReceiveNodeServiceThread.Create(servMan_, proxy_, port_, tableMan_.getNodeTable(), logger_);
end;

end.
