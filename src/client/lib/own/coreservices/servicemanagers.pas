unit servicemanagers;
 {
   TServiceManager handles all services available to the GPU core.servicemanagers

   (c) by 2010 HB9TVM and the GPU Team
}

interface

uses
  downloadthreadmanagers, servermanagers, dbtablemanagers, loggers,
  receivenodeservices;

type TServiceManager = class(TObject)
   public
    constructor Create(downMan : TDownloadThreadManager; servMan : TServerManager;
                       tableMan : TDbTableManager; logger : TLogger);
    destructor Destroy;

    function getReceivenodeservice() : TReceiveNodeService;

   private
     receivenodeservice_ : TReceiveNodeService;

     downMan_  : TDownloadThreadManager;
     servMan_  : TServerManager;
     tableMan_ : TDbTableManager;
     logger_   : TLogger

end;

implementation

constructor TServiceManager.Create(downMan : TDownloadThreadManager; servMan : TServerManager;
                                   tableMan : TDbTableManager; logger : TLogger);
begin
 downMan_  := downMan;
 servMan_  := servMan;
 tableMan_ := tableMan_;
 logger_   := logger_;

 receivenodeservice_ := TReceiveNodeService.Create(downMan_, servMan_, tableMan_.getNodeTable(); logger_);
end;

destructor TServiceManager.Destroy;
begin
  receivenodeservice_.Free;
end;

function TServiceManager.getReceivenodeservice() : TReceiveNodeService;
begin
 Result := receivenodeservice_;
end;
end.
