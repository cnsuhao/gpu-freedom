unit coreservices;
{

  This unit sets up the ancestor for each request and response done to the GPU II servers

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses
  managedthreads, servermanagers, loggers;

type TCoreServiceThread = class(TManagedThread)
  public
    constructor Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger);

  protected
    servMan_ : TServerManager;
    logger_  : TLogger;
    proxy_,
    port_    : String;
end;


type TReceiveServiceThread = class(TCoreServiceThread)
end;

type TTransmitServiceThread = class(TCoreServiceThread)
end;


implementation

constructor TCoreServiceThread.Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger);
begin
  inherited Create(true); // suspended
  servMan_ := servMan;
  logger_  := logger;
  proxy_   := proxy;
  port_    := port;
end;


end.
