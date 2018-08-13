let fileContent
let userChoices

const checkBox = (name) => {
    const inputs = document.getElementsByName(name).values()
    var arr = []
    for (let value of inputs) {
        isChecked = value.checked
        if (isChecked) arr.push(value.dataset.column)
    }
    return arr
}

async function handleSubmit() {
    userChoices = {
        'categorical': checkBox('categorical'),
        'datetime': checkBox('datetime'),
        'drop': checkBox('drop'),
        'y': checkBox('y'),
        'drop_rest': document.getElementById('drop_rest').checked,
        'supervised': document.getElementById('supervised').checked,
    }

    const putObject = { 'data': fileContent, 'param': userChoices }
    const results = await window.fetch(
        '/config',
        { method: 'POST', body: JSON.stringify(putObject) }
    )
    console.log(results)
}

function handleFileUpload(evt) {
    const file = evt.target.files[0]
    const container = document.getElementById('container')
    if (!file) return
    const reader = new FileReader()
    reader.onload = e => {
        fileContent = e.target.result
        const [headers, ...values] = fileContent.split('\n')
        const columns = headers.split(',').map(_.kebabCase)
        columns.forEach(column => {
            const column_table = document.getElementById('table')
            column_table.insertAdjacentHTML('beforeend',
                `
                <tr>
                    <td href='column'>${column}</td>
                    <td><input data-column=${column} type='checkbox' name='categorical' value='True' /></td>
                    <td><input data-column=${column} type='checkbox' name='datetime' value = 'True' /></td>
                    <td><input data-column=${column} type='checkbox' name='drop' value = 'True' /></td>
                    <td><input data-column=${column} type='checkbox' name='y' value = 'True' /></td>
                </tr>
                `
            )
        })
    }
    reader.readAsText(file)
}

document.addEventListener("DOMContentLoaded", async function () {
    const fileInput = document.getElementById('file_input')
    fileInput.onchange = handleFileUpload
    const submit = document.getElementById('submit')
    submit.onclick = handleSubmit
})
