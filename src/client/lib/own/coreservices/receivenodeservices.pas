unit receivenodeservices;
{

  This unit receives a list of active XML nodes from GPU II servers
   and stores it in the TDbNodeTable object.receivenodeservices

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, downloadthreadmanagers, servermanagers, nodetable;

type TReceiveNodeService = class(TReceiveService)
 public
  constructor Create(downMan : TDownloadThreadManager; servMan : TServerManager;
                     appPath: String; fnodetable : TDbNodeTable);

  procedure receive(); virtual; override;

  procedure onError : TDownloadFinishedEvent;
  procedure onFinished : TDownloadFinishedEvent;
 private
   nodetable_ : TDbNodeTable;
end;

implementation

constructor TReceiveNodeService.Create(downMan : TDownloadThreadManager; servMan : TServerManager;
                                       appPath : String; fnodetable : TDbNodeTable);
begin
 inherited Create(downMan, servMan, appPath);
 nodetable_ := fnodetable;
end;

procedure TReceiveNodeService.receive(); virtual; override;
begin
  if not enabled_ then Exit;
  enabled_ := false;

  downMan_.download(servMan_.getServerUrl()+'/list_computers_online_xml.php', self.onFinished, self.onError);
end;

procedure onError : TDownloadFinishedEvent;
begin
 enabled_ := true;
end;

procedure onFinished : TDownloadFinishedEvent;
begin
 enabled_ := true;
end;

end;
