unit coreservices;
{

  This unit sets up the ancestor for each request and response done to the GPU II servers

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses
  managedthreads, servermanagers, loggers, downloadutils, XMLRead, DOM, Classes, SysUtils;

type TCoreServiceThread = class(TManagedThread)
  public
    constructor Create(var logger : TLogger);

  protected
    logger_  : TLogger;
end;

type TCommServiceThread = class(TCoreServiceThread)
  public
    constructor Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger);

  protected
    servMan_ : TServerManager;
    proxy_,
    port_    : String;
    procedure finishComm(url, logHeader, logSuccess : String);

end;


type TInternalServiceThread = class(TCoreServiceThread)
end;

type TTransmitServiceThread = class(TCommServiceThread)
   procedure transmit(url, request, logHeader : String; noargs : Boolean);
   procedure finishTransmit(url, logHeader, logSuccess : String);
end;


type TReceiveServiceThread = class(TCommServiceThread)
   procedure receive(url, request, logHeader : String; var xmldoc : TXmlDocument; noargs : Boolean);
   procedure finishReceive(url, logHeader, logSuccess : String; var xmldoc : TXmlDocument);
end;



implementation

constructor TCoreServiceThread.Create(var logger : TLogger);
begin
  inherited Create(true); // suspended
  logger_  := logger;
end;

constructor TCommServiceThread.Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger);
begin
  inherited Create(logger);
  servMan_ := servMan;
  proxy_   := proxy;
  port_    := port;
end;


procedure TTransmitServiceThread.transmit(url, request, logHeader : String; noargs : Boolean);
var
    stream    : TMemoryStream;
begin
 stream  := TMemoryStream.Create;
 erroneous_ := not downloadToStream(url+request+getProxyArg(noargs),
               proxy_, port_, logHeader, logger_, stream);

 if stream <>nil then stream.Free  else logger_.log(LVL_SEVERE,
         logHeader+'Internal error in coreservices.pas, stream is nil');
 done_ := true;
end;


procedure TReceiveServiceThread.receive(url, request, logHeader : String; var xmldoc : TXmlDocument; noargs : Boolean);
var
    stream    : TMemoryStream;
begin
 xmldoc := TXMLDocument.Create();
 stream  := TMemoryStream.Create;
 erroneous_ := not downloadToStream(url+request+getProxyArg(noargs),
               proxy_, port_, logHeader, logger_, stream);

 if stream=nil then
  begin
   logger_.log(LVL_SEVERE, logHeader+'Internal error in coreservices.pas, stream is nil');
   erroneous_ := true;
  end;

 if not erroneous_ then
 begin
  try
    stream.Position := 0; // to avoid Document root is missing exception
    ReadXMLFile(xmldoc, stream);
  except
     on E : Exception do
        begin
           erroneous_ := true;
           logger_.log(LVL_SEVERE, logHeader+'Exception catched in Execute: '+E.Message);
        end;
  end; // except
  end; // if not erroneous
  if stream<>nil then stream.Free;
end;

procedure TCommServiceThread.finishComm(url, logHeader, logSuccess : String);
begin
 if erroneous_ then
    begin
     servMan_.increaseFailures(url);
     logger_.log(LVL_SEVERE, logHeader+'Service finished but ERRONEOUS flag set :-(')
    end
 else
   logger_.log(LVL_INFO, logHeader+logSuccess);
 done_ := true;
end;


procedure TReceiveServiceThread.finishReceive(url, logHeader, logSuccess : String; var xmldoc : TXmlDocument);
begin
 if xmldoc=nil then
   begin
    logger_.log(LVL_SEVERE, logHeader+'xmldoc is nil in finalize');
    erroneous_ := true;
   end
 else
    xmldoc.Free;

 finishComm(url, logHeader, logSuccess);
end;

procedure TTransmitServiceThread.finishTransmit(url, logHeader, logSuccess : String);
begin
 finishComm(url, logHeader, logSuccess);
end;


end.
