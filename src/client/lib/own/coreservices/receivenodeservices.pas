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

   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveNodeServiceThread.Create(servMan : TServerManager; proxy, port : String;
                                       fnodetable : TDbNodeTable; logger : TLogger);
begin
 inherited Create(servMan, proxy, port, logger);
 nodetable_ := fnodetable;
end;


procedure TReceiveNodeServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbnode : TDbNodeRow;
    node   : TDOMNode;
    port   : String;
begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
        try
             begin
               dbnode.nodeid :=node.FindNode('nodeid').TextContent;
               //TODO: dbnode.defaultserver_id
               dbnode.nodename :=node.FindNode('processor').TextContent;
               dbnode.country :=node.FindNode('country').TextContent;
               dbnode.region :=node.FindNode('region').TextContent;
               dbnode.ip :=node.FindNode('ip').TextContent;
               port := node.FindNode('port').TextContent;
               if port='' then port:='0';
               dbnode.port :=StrToInt(port);
               //TODO: EAccessViolation dbnode.localip :=node.FindNode('localip').TextContent;
               dbnode.os :=node.FindNode('operatingsystem').TextContent;
               dbnode.cputype :=node.FindNode('cputype').TextContent;
               dbnode.version :=node.FindNode('version').TextContent;
               dbnode.acceptincoming :=(node.FindNode('accept').TextContent='1');
               dbnode.gigaflops :=StrToInt(node.FindNode('speed').TextContent);
               dbnode.ram :=StrToInt(node.FindNode('ram').TextContent);
               dbnode.mhz :=StrToInt(node.FindNode('mhz').TextContent);
               //dbnode.nbcpus :=StrToInt(node.FindNode('cpus').TextContent); empty
               dbnode.online := true; //TODO: check if this is correct
               dbnode.uptime :=StrToFloatDef(node.FindNode('uptime').TextContent, 0);
               dbnode.totaluptime :=StrToFloatDef(node.FindNode('totuptime').TextContent, 0);
               dbnode.longitude :=StrToFloatDef(node.FindNode('geolocation_x').TextContent, 0);
               dbnode.latitude :=StrToFloatDef(node.FindNode('geolocation_y').TextContent, 0);
               nodetable_.insertOrUpdate(dbnode);
               logger_.log(LVL_DEBUG, 'Added <'+dbnode.nodename+'> to tbnode table.');
             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, '[TReceiveNodeServiceThread]> Exception catched in parseXML: '+E.Message);
              end;
          end; // except

       node := node.NextSibling;
     end;  // while Assigned(node)

   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;

procedure TReceiveNodeServiceThread.Execute;
var xmldoc    : TXMLDocument;
    stream    : TMemoryStream;
    proxyseed : String;
begin
 stream  := TMemoryStream.Create;

 proxyseed  := getProxySeed;
 logger_.log(LVL_DEBUG, 'Proxy seed is: '+proxyseed);
 erroneous_ := not downloadToStream(servMan_.getServerUrl()+'/list_computers_online_xml.php?randomseed='+proxyseed,
               proxy_, port_, '[TReceiveNodeServiceThread]> ', logger_, stream);

 if not erroneous_ then
 begin
  try
    stream.Position := 0; // to avoid Document root is missing exception
    xmldoc := TXMLDocument.Create();
    ReadXMLFile(xmldoc, stream);
  except
     on E : Exception do
        begin
           erroneous_ := true;
           logger_.log(LVL_SEVERE, '[TReceiveNodeServiceThread]> Exception catched in Execute: '+E.Message);
        end;
  end; // except

  if not erroneous_ then parseXml(xmldoc);
  xmldoc.Free;
 end;


 if stream <>nil then stream.Free  else logger_.log(LVL_SEVERE, '[TReceiveNodeServiceThread]> Internal error in receivenodeservices.pas, stream is nil');

 done_ := true;
end;

end.
