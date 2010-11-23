unit receivenodeservices;
{

  This unit receives a list of active XML nodes from GPU II servers
   and stores it in the TDbNodeTable object.receivenodeservices

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, downloadthreadmanagers, servermanagers, nodetable, loggers,
     XMLRead, DOM;

type TReceiveNodeService = class(TReceiveService)
 public
  constructor Create(downMan : TDownloadThreadManager; servMan : TServerManager;
                     fnodetable : TDbNodeTable; logger : TLogger);

  procedure receive(); virtual; override;

  procedure onError : TDownloadFinishedEvent;
  procedure onFinished : TDownloadFinishedEvent;
 private
   nodetable_ : TDbNodeTable;
end;

implementation

constructor TReceiveNodeService.Create(downMan : TDownloadThreadManager; servMan : TServerManager;
                                       fnodetable : TDbNodeTable; logger : TLogger);
begin
 inherited Create(downMan, servMan, logger);
 nodetable_ := fnodetable;
end;

procedure TReceiveNodeService.receive(); virtual; override;
begin
  if not enabled_ then Exit;
  enabled_ := false;

  if downMan_.download(servMan_.getServerUrl()+'/list_computers_online_xml.php', self.onFinished, self.onError) = -1 then
      begin
        enabled_ := true;
        Exit;
      end;
end;

procedure onError : TDownloadFinishedEvent;
begin
 enabled_ := true;
end;


procedure onFinished : TDownloadFinishedEvent;
var xmldoc : TXMLDocument;
    nodes,
    node   : TDOMNode;
    j      : Longint;
    dbnode : TDbNodeRow;
begin
 ReadXML(xmldoc, stream);

 nodes := xmldoc.DocumentElement.FirstChild;
 logger_.log(LVL_DEBUG, 'Parsing of XML started...');
 if Assigned(nodes) then
    begin
        try
          for j := 0 to (nodes.ChildNodes.Count - 1) do
             begin
               node := nodes.ChildNodes.Item[i];

               dbnode.nodeid :=node.FindNode('nodeid').TextContent;
               //TODO: dbnode.defaultserver_id
               dbnode.nodename :=node.FindNode('processor').TextContent;
               dbnode.country :=node.FindNode('country').TextContent;
               dbnode.region :=node.FindNode('region').TextContent;
               dbnode.ip :=node.FindNode('ip').TextContent;
               dbnode.port :=node.FindNode('port').TextContent;
               dbnode.localip :=node.FindNode('localip').TextContent;
               dbnode.os :=node.FindNode('os').TextContent;
               dbnode.cputype :=node.FindNode('cputype').TextContent;
               dbnode.version :=node.FindNode('version').TextContent;
               dbnode.acceptincoming :=node.FindNode('acceptincoming').TextContent;
               dbnode.gigaflops :=node.FindNode('gigaflops').TextContent;
               dbnode.ram :=node.FindNode('ram').TextContent;
               dbnode.mhz :=node.FindNode('mhz').TextContent;
               dbnode.nbcpus :=node.FindNode('nbcpus').TextContent;
               dbnode.online := true; //TODO: check if this is correct
               dbnode.uptime :=node.FindNode('uptime').TextContent;
               dbnode.totaluptime :=node.FindNode('totaluptime').TextContent;
               dbnode.longitude :=node.FindNode('longitude').TextContent;
               dbnode.latitude :=node.FindNode('latitude').TextContent;

               nodetable_.insertOrUpdate(dbnode);
               logger_.log(LVL_DEBUG, 'Added <'+dbnode.nodename+'> to tbnode table.');
             end;
          except
           on E : Exception do
              begin
                logger_.log(LVL_SEVERE, 'Exception catched: '+E.Message);
              end;
    end;  // if

 enabled_ := true;
 logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;

end;
