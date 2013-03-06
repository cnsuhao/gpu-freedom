unit receiveparamservices;
{

  This unit receives parameters from a GPU II server, stores them into local
  tbparameter table and into myConfId structure.

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}

interface

uses coreservices, servermanagers, dbtablemanagers,
     loggers, coreconfigurations, parametertables,
     Classes, SysUtils, DOM, identities;

type TReceiveParamServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager);

 protected
   procedure Execute; override;

 private
   procedure parseXml(var xmldoc : TXMLDocument);

end;



implementation

constructor TReceiveParamServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                                              var conf : TCoreConfiguration; var tableman : TDbTableManager);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TReceiveParamServiceThread]> ', conf, tableman);
end;

procedure TReceiveParamServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    node     : TDOMNode;
    dbnode   : TDbParameterRow;
begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
        try
             begin
               dbnode.paramtype         := node.FindNode('paramtype').TextContent;
               dbnode.paramname         := node.FindNode('paramname').TextContent;
               dbnode.paramvalue        := node.FindNode('paramvalue').TextContent;
               dbnode.create_dt         := Now();
               dbnode.update_dt         := Now(); //TODO parse from string from server

               logger_.log(LVL_DEBUG, logHeader_+'Adding or updating parameter '+dbnode.paramname+' to tbparameter table.');
               tableman_.getParameterTable().insertorupdate(dbnode);
               logger_.log(LVL_DEBUG, 'record count: '+IntToStr(tableman_.getParameterTable().getDS().RecordCount));

             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, logHeader_+'Exception catched in parseXML: '+E.Message);
              end;
          end; // except

        node := node.NextSibling;
     end;  // while Assigned(param)

   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
   logger_.log(LVL_DEBUG, 'All parameters updated succesfully');
end;

{
TODO: load this parameters from tbparameter into myConfId
with myConfId do
               begin

                 receive_servers_each  := StrToInt(param.FindNode('receive_servers_each').TextContent);
                 receive_nodes_each    := StrToInt(param.FindNode('receive_nodes_each').TextContent);
                 transmit_node_each    := StrToInt(param.FindNode('transmit_node_each').TextContent);
                 receive_jobs_each     := StrToInt(param.FindNode('receive_jobs_each').TextContent);
                 transmit_jobs_each    := StrToInt(param.FindNode('transmit_jobs_each').TextContent);
                 receive_channels_each := StrToInt(param.FindNode('receive_channels_each').TextContent);
                 transmit_channels_each:= StrToInt(param.FindNode('transmit_channels_each').TextContent);
                 receive_chat_each     := StrToInt(param.FindNode('receive_chat_each').TextContent);
                 purge_server_after_failures := StrToInt(param.FindNode('purge_server_after_failures').TextContent);
end; // with
}

procedure TReceiveParamServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
 receive('/supercluster/list_parameters.php?xml=1&paramtype=CLIENT', xmldoc, false);

 if not erroneous_ then
     parseXml(xmldoc);

 finishReceive('Service retrieved global parameters from superserver succesfully :-)', xmldoc);
end;




end.
