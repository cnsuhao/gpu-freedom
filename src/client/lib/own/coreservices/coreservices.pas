unit coreservices;
{

  This unit sets up the ancestor for each request and response done to the GPU II servers

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses
 downloadthreadmanagers, servermanagers;

type TCoreService = class(TObject)
  public
    constructor Create(downMan : TDownloadThreadManager; servMan : TServerManager; logger : TLogger);

    function isEnabled : Boolean;

  protected
    enabled_ : Boolean;
    downMan_ : TDownloadThreadManager;
    servMan_ : TServerManager;
    logger_  : TLogger;
end;


type TReceiveService = class(TCoreService)
  procedure receive(); virtual; abstract;
end;

type TTransmitService = class(TCoreService)
  procedure transmit(); virtual; abstract;
end;


implementation

constructor TCoreService.Create(downMan : TDownloadThreadManager; servMan : TServerManager; logger : TLogger);
begin
  inherited Create;
  enabled_ := true;
  downMan_ := downMan;
  servMan_ := servMan;
  logger_  := logger;
end;


function TCoreService.isEnabled : Boolean;
begin
 Result := enabled_;
end;


end.
