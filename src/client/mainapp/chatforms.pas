unit chatforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  ExtCtrls, StdCtrls;

type

  { TChatForm }

  TChatForm = class(TForm)
    btnSend: TButton;
    cbSelectChannel: TComboBox;
    mmChat: TMemo;
    mmSubmitChat: TMemo;
    PanelBottom: TPanel;
    PanelTop: TPanel;
  private
    { private declarations }
  public
    { public declarations }
  end; 

var
  ChatForm: TChatForm;

implementation

initialization
  {$I chatforms.lrs}

end.

