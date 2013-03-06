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
   procedure loadParametersIntoConfiguration;
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
   loadParametersIntoConfiguration;
   logger_.log(LVL_DEBUG, 'All parameters updated succesfully');
end;


procedure TReceiveParamServiceThread.loadParametersIntoConfiguration;
begin
    with myConfId do
                begin
                  receive_servers_each  := StrToInt(tableman_.getParameterTable().getParameter('CLIENT', 'receive_servers_each','3553'));
                  receive_nodes_each    := StrToInt(tableman_.getParameterTable().getParameter('CLIENT', 'receive_nodes_each','120'));
                  transmit_node_each    := StrToInt(tableman_.getParameterTable().getParameter('CLIENT', 'transmit_node_each','121'));
                  receive_jobs_each     := StrToInt(tableman_.getParameterTable().getParameter('CLIENT', 'receive_jobs_each','60'));
                  transmit_jobs_each    := StrToInt(tableman_.getParameterTable().getParameter('CLIENT', 'transmit_jobs_each','60'));
                  receive_channels_each := StrToInt(tableman_.getParameterTable().getParameter('CLIENT', 'receive_channels_each','60'));
                  transmit_channels_each:= StrToInt(tableman_.getParameterTable().getParameter('CLIENT', 'transmit_channels_each','60'));
                  receive_chat_each     := StrToInt(tableman_.getParameterTable().getParameter('CLIENT', 'receive_chat_each','60'));
                  purge_server_after_failures := StrToInt(tableman_.getParameterTable().getParameter('CLIENT', 'purge_server_after_failures','30'));
    end; // with

 logger_.log(LVL_DEBUG, 'receinve_nodes_each is set to '+IntToStr(myConfID.receive_nodes_each));
end;

procedure TReceiveParamServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
 receive('/supercluster/list_parameters.php?xml=1&paramtype=CLIENT', xmldoc, false);

 if not erroneous_ then
     parseXml(xmldoc);

 finishReceive('Service retrieved global parameters from superserver succesfully :-)', xmldoc);
end;




end.
