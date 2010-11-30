unit receivenodeservices;
{

  This unit receives a list of active XML nodes from GPU II servers
   and stores it in the TDbNodeTable object.receivenodeservices

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers,
     nodetable, loggers, downloadutils,
     XMLRead, DOM, Classes, SysUtils;

type TReceiveNodeServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(servMan : TServerManager; proxy, port : String;
                     fnodetable : TDbNodeTable; logger : TLogger);
 protected
    procedure Execute; override;

 private
   nodetable_ : TDbNodeTable;
end;

implementation

constructor TReceiveNodeServiceThread.Create(servMan : TServerManager; proxy, port : String;
                                       fnodetable : TDbNodeTable; logger : TLogger);
begin
 inherited Create(servMan, proxy, port, logger);
 nodetable_ := fnodetable;
end;


procedure TReceiveNodeServiceThread.Execute;
var xmldoc : TXMLDocument;
    nodes,
    node   : TDOMNode;
    j      : Longint;
    dbnode : TDbNodeRow;
    stream : TMemoryStream;
    port   : String;
begin
 erroneous_ := downloadToStream(servMan_.getServerUrl()+'/list_computers_online_xml.php',
               proxy_, port_, '[TReceiveNodeServiceThread]> ', logger_, stream);
 if erroneous_ then
      begin
        done_ := true;
      end
 else
 begin
 ReadXMLFile(xmldoc, stream);

 nodes := xmldoc.DocumentElement.FirstChild;
 logger_.log(LVL_DEBUG, 'Parsing of XML started...');
 if Assigned(nodes) then
    begin
        try
          for j := 0 to (nodes.ChildNodes.Count - 1) do
             begin
               node := nodes.ChildNodes.Item[j];

               dbnode.nodeid :=node.FindNode('nodeid').TextContent;
               //TODO: dbnode.defaultserver_id
               dbnode.nodename :=node.FindNode('processor').TextContent;
               dbnode.country :=node.FindNode('country').TextContent;
               dbnode.region :=node.FindNode('region').TextContent;
               dbnode.ip :=node.FindNode('ip').TextContent;
               port := node.FindNode('port').TextContent;
               if port='' then port:='0';
               dbnode.port :=StrToInt(port);
               dbnode.localip :=node.FindNode('localip').TextContent;
               dbnode.os :=node.FindNode('os').TextContent;
               dbnode.cputype :=node.FindNode('cputype').TextContent;
               dbnode.version :=node.FindNode('version').TextContent;
               dbnode.acceptincoming :=(node.FindNode('acceptincoming').TextContent='true');
               dbnode.gigaflops :=StrToInt(node.FindNode('gigaflops').TextContent);
               dbnode.ram :=StrToInt(node.FindNode('ram').TextContent);
               dbnode.mhz :=StrToInt(node.FindNode('mhz').TextContent);
               dbnode.nbcpus :=StrToInt(node.FindNode('nbcpus').TextContent);
               dbnode.online := true; //TODO: check if this is correct
               dbnode.uptime :=StrToFloatDef(node.FindNode('uptime').TextContent, 0);
               dbnode.totaluptime :=StrToFloatDef(node.FindNode('totaluptime').TextContent, 0);
               dbnode.longitude :=StrToFloatDef(node.FindNode('longitude').TextContent, 0);
               dbnode.latitude :=StrToFloatDef(node.FindNode('latitude').TextContent, 0);

               nodetable_.insertOrUpdate(dbnode);
               logger_.log(LVL_DEBUG, 'Added <'+dbnode.nodename+'> to tbnode table.');
             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, 'Exception catched: '+E.Message);
              end;
          end; // except
    end;  // if

 done_ := true;
 logger_.log(LVL_DEBUG, 'Parsing of XML over.');
 end; // erroneous = false
end;

end.
