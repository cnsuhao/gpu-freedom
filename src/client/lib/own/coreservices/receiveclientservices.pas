unit receiveclientservices;
{

  This unit receives a list of active XML nodes from GPU II servers
   and stores it in the TDbClientTable object.receivenodeservices

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers, coreconfigurations, clienttables,
     dbtablemanagers, identities, loggers, Classes, SysUtils, DOM, synacode;

type TReceiveClientServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager);
 protected
    procedure Execute; override;

 private
   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveClientServiceThread.Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager);
begin
 inherited Create(servMan, proxy, port, logger, '[TReceiveClientServiceThread]> ', conf, tableman);
end;


procedure TReceiveClientServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbnode   : TDbClientRow;
    node     : TDOMNode;
begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
 try
  begin
    node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
               dbnode.nodeid            := node.FindNode('nodeid').TextContent;
               dbnode.server_id         := srv_.id;
               dbnode.nodename          := node.FindNode('nodename').TextContent;
               dbnode.country           := node.FindNode('country').TextContent;
               dbnode.region            := node.FindNode('region').TextContent;
               dbnode.city              := node.FindNode('city').TextContent;
               dbnode.zip               := node.FindNode('zip').TextContent;
               dbnode.description       := node.FindNode('description').TextContent;
               dbnode.ip                := ''; // not transmitted
               dbnode.port              := ''; // not transmitted
               dbnode.localip           := ''; // not transmitted
               dbnode.os                := node.FindNode('os').TextContent;
               dbnode.cputype           := ''; // not transmitted
               dbnode.version           := StrToFloatDef(node.FindNode('version').TextContent, -1);
               dbnode.acceptincoming    := (node.FindNode('acceptincoming').TextContent='true');
               dbnode.gigaflops    := StrToInt(node.FindNode('gigaflops').TextContent);
               dbnode.ram          := StrToInt(node.FindNode('ram').TextContent);
               dbnode.mhz          := StrToInt(node.FindNode('mhz').TextContent);
               dbnode.nbcpus       := StrToInt(node.FindNode('nbcpus').TextContent);
               dbnode.bits         := StrToInt(node.FindNode('bits').TextContent);
               dbnode.online  := true;
               dbnode.updated := true;
               dbnode.uptime      := StrToFloatDef(node.FindNode('uptime').TextContent, 0);
               dbnode.totaluptime := StrToFloatDef(node.FindNode('totaluptime').TextContent, 0);
               dbnode.longitude   := StrToFloatDef(node.FindNode('longitude').TextContent, 0);
               dbnode.latitude    := StrToFloatDef(node.FindNode('latitude').TextContent, 0);
               dbnode.userid      := ''; // not transmittted
               dbnode.team        := node.FindNode('team').TextContent;
               tableman_.getClientTable().insertOrUpdate(dbnode);
               logger_.log(LVL_DEBUG, logHeader_+'Updated or added <'+dbnode.nodename+'> to tbclient table.');
       node := node.NextSibling;
     end;  // while Assigned(node)

 end; // try
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, logHeader_+'Exception catched in parseXML: '+E.Message);
              end;
          end; // except



   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;

procedure TReceiveClientServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
 receive('/cluster/list_clients.php?xml=1&nodeid='+encodeUrl(myGPUID.NodeId), xmldoc, false);

 if not erroneous_ then
    begin
     tableMan_.getClientTable().execSQL('UPDATE tbclient set updated=0;');
     parseXml(xmldoc);
     if not erroneous_ then
        tableMan_.getClientTable().execSQL('UPDATE tbclient set online=updated;');
    end;

 finishReceive('Service updated table TBCLIENT succesfully :-)', xmldoc);
end;

end.
