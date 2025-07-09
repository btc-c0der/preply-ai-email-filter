# Configurações SMTP/IMAP para Email Filter App

## Configurações por Provedor

### Gmail
- **Servidor:** imap.gmail.com
- **Porta:** 993 (SSL)
- **Configuração:** Ativar autenticação de 2 fatores e criar App Password
- **URL App Password:** https://myaccount.google.com/apppasswords

### Outlook/Hotmail
- **Servidor:** outlook.office365.com
- **Porta:** 993 (SSL)
- **Configuração:** Usar senha normal ou App Password

### Yahoo Mail
- **Servidor:** imap.mail.yahoo.com
- **Porta:** 993 (SSL)
- **Configuração:** Ativar autenticação de 2 fatores e criar App Password
- **URL App Password:** https://login.yahoo.com/account/security

### iCloud Mail
- **Servidor:** imap.mail.me.com
- **Porta:** 993 (SSL)
- **Configuração:** Ativar autenticação de 2 fatores e criar App Password
- **URL App Password:** https://appleid.apple.com/

## Instruções de Uso

### 1. Configuração de App Password (Recomendado)

Para maior segurança, use App Passwords em vez de senhas normais:

#### Gmail:
1. Ative a autenticação de 2 fatores
2. Vá para https://myaccount.google.com/apppasswords
3. Gere uma senha específica para este app
4. Use esta senha no campo "Senha/App Password"

#### Yahoo:
1. Ative a autenticação de 2 fatores
2. Vá para https://login.yahoo.com/account/security
3. Gere uma senha de app
4. Use esta senha no campo "Senha/App Password"

### 2. Teste de Conexão

Sempre use o botão "Testar Conexão" antes de conectar:
- Verifica se as credenciais estão corretas
- Testa a conectividade com o servidor
- Valida as configurações SSL

### 3. Download de Emails

- **Pasta:** Selecione a pasta (INBOX é padrão)
- **Limite:** Número máximo de emails para baixar
- **Dias:** Quantos dias para trás buscar emails

### 4. Classificação Automática

Os emails baixados são automaticamente classificados se um modelo estiver carregado:
- **Relevantes:** Emails importantes ou pessoais
- **Não Relevantes:** Spam ou emails promocionais
- **Categorias:** importante, promocional, spam, pessoal, outros

## Solução de Problemas

### Erro de Autenticação
- Verifique username/email
- Use App Password em vez de senha normal
- Confirme que 2FA está ativo (Gmail/Yahoo)

### Erro de Conexão
- Verifique servidor e porta
- Confirme conexão com internet
- Tente desabilitar firewall temporariamente

### Erro SSL
- Confirme que SSL está habilitado
- Verifique se o servidor suporta SSL na porta especificada

### Nenhum Email Encontrado
- Aumente o número de dias para busca
- Verifique se a pasta selecionada tem emails
- Confirme que o filtro de data está correto

## Segurança

### Boas Práticas:
1. **Use App Passwords** em vez de senhas principais
2. **Não compartilhe** credenciais
3. **Revogue App Passwords** quando não precisar mais
4. **Use conexões SSL** sempre que possível
5. **Monitore** atividade de login suspeita

### Dados Locais:
- Emails são processados localmente
- Não são enviados para servidores externos
- Credenciais não são armazenadas
- Modelo de IA roda na máquina local

## Limitações

- Máximo 500 emails por download (para performance)
- Apenas texto simples (HTML é convertido)
- Suporte limitado a anexos
- Funciona apenas com protocolos IMAP

## Proveedores Testados

✅ **Gmail** - Funcional com App Password
✅ **Outlook/Office365** - Funcional  
✅ **Yahoo Mail** - Funcional com App Password
✅ **iCloud** - Funcional com App Password
⚠️ **Outros** - Use configuração personalizada
