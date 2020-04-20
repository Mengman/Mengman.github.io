---
layout: post
title: "软件设计模型-功能开关"
date: 2020-04-20 17:26:21 +0800
categories: designpattern softwareengineering
---

## 功能开关
本文受到 Pete Hodgson 的文章 [Feature Toggles (aka Feature Flags)](https://www.martinfowler.com/articles/feature-toggles.html) 启发，原文内容比本文更加丰富。

### 简单的功能开关

场景一：你刚刚开发完一个新的功能，然后需求告诉你，由于这个功能使用场景的限制，需要添加一个开关，能够让这个功能在某些环境打开或者关闭。

场景二：你实现了一个新版本的算法，但是并不希望立刻在线上启用，希望能进行更多的测试，所以需要添加一个开关，在测试环境打开，在生产环境保持关闭。

这个时候你只需要实现一个简单的功能开关即可。

```go
func doSomeThing() {
    useNewAlgorithm := false
    // useNewAlgorithm := true // uncomment if you working with new algorithm
    if useNewAlgoritem {
        doSomeThingNew()
    } else {
        doSomeThingOld()
    }
}

func doSomeThingNew() {
    // implement something new
}

func doSomeThingOld() {
    // implement something old
}

```

我们把新的功能写到 <code>doSomeThingNew</code> 函数中，并通过 <code>useNewAlgorithm</code> 变量来控制是否调用新功能。这个方法虽然简单，但是我们无法在运行时动态的去决定是否启用新的功能。为了实现对于新功能的动态控制，我们可以引入一个**开关路由器**（toggle router）来实现。

### 开关路由器

```go
type ToggleRouter struct {
	featureConfig map[string]struct{}
}

func (tr *ToggleRouter) SetFeature(featureName string, isEnabled bool) {
	if isEnabled {
		tr.featureConfig[featureName] = struct{}{}
	} else {
		delete(tr.featureConfig, featureName)
	}
}

func (tr *ToggleRouter) IsFeatureEnabled(featureName string) bool {
	_, ok := tr.featureConfig[featureName]
	return ok
}

func NewToggleRouter(features []string) (tr *ToggleRouter) {
	tr = &ToggleRouter{
		featureConfig: make(map[string]struct{}),
	}

	for _, feat := range features {
		tr.SetFeature(feat, true)
	}

	return
}

var toggleRouter = NewToggleRouter([]string{"newAlgo"})

func doSomeThing() {
    if toggleRouter.IsFeatureEnabled("newAlgo") {
        doSomeThingNew()
    } else {
        doSomeThingOld()
    }
}
```

通过开关路由器我们可以动态的设置功能的启用与否。ToggleRouter 可以通过配置文件创建，或者通过配置界面，或者 API 接口来进行配置。到现在问题似乎已经解决了，然而随着越来越多的新功能加入，越来越多的功能需要通过开关来配置是否开启，代码中出现了越来越多的**开关点**，代码也变得越来越难以理解和维护。

让我们看一个新例子，<code>InvoiceEmailler</code> 会根据收据发送一封电子邮件，现在我们希望能加上一个新的功能，在邮件中添加退货链接。而这个退货链接只是在特定的情况下才会触发。下面代码中是根据 开关中是否包含 <code>next-gen-ecomm</code> 来判断的。

```go
var toggle = GetFeatureToggleRouter()

type InvoiceEmailler struct {
    invoice Invoice
}

func (emailler *InvoiceEmailler) GenerateInvoiceEmail() Email {
    baseEmail := buildEmailForInvoice(emailler.invoice)
    if toggle.IsFeatureEnabled("next-gen-ecomm") {
        return addOrderCallationContentToEmail(baseEmail)
    } else {
        return baseEmail
    }
}
```

但是，对于一个功能是否启用的判断逻辑可能是十分复杂的，也许在 A 情况下需要开启，在 B 情况下需要关闭；或者需要对某一部分用户开启这一项功能，对另一部分用户关闭这一功能；开关的判断逻辑可能会频繁的迭代。如果要更新判断逻辑，我们不得不在每个开关点去修改判断逻辑。这是代码中的第一个问题：**开关点与开关的判断逻辑耦合在一起**。

### 解耦开关点与开关路由器

幸好[“软件中的任何问题都可以通过引入一个间接层来解决”](https://en.wikipedia.org/wiki/Fundamental_theorem_of_software_engineering), 通过增加一个**开关逻辑判断层**我们可以解耦开关点与开关判断逻辑。

```go
type FeatureDecisions struct {
    toggle *ToggleRouter
}

func (descisions *FeatureDecisions) IncludeOrderCancellationInEmail() bool {
    return descisions.toggle.IsFeatureEnabled("new-gen-ecomm")
}

func NewFeatureDecisions(toggle *ToggleRouter) FeatureDecisions {
    return FeatureDecisions{toggle: toggle}
}


var toggle = GetFeatureToggleRouter()
var featureDecisions = NewFeatureDecision(toggle)

type InvoiceEmailler struct {
    invoice Invoice
}

func (emailler *InvoiceEmailler) GenerateInvoiceEmail() Email {
    baseEmail := buildEmailForInvoice(emailler.invoice)
    if featureDecisions.IncludeOrderCancellationInEmail() {
        return addOrderCallationContentToEmail(baseEmail)
    } else {
        return baseEmail
    }
}

```

通过引入 <code>FeatureDecisions</code> 之后，<code>InvoiceEmailler</code> 不再关心添加退货链接的逻辑是什么；而后续对与添加退货链接逻辑的更新也与 <code>InvoiceEmailler</code> 无关。从而实现了**开关点与开关判断逻辑的解耦**。

### 判断翻转(Inversion of Decision)

虽然在上一步中，我们通过添加**判断逻辑层**实现了判断逻辑与开关点的解耦，但是 <code>InvoiceEmailler</code> 依旧与 <code>FeatureDecisions</code> 耦合在一起，在执行 <code>GenerateInvoiceEmail()</code> 需要先创建或者获取 <code>FeatureDecisions</code> ，这处代码“坏味道”带来了两个问题：

1. 它不方便对代码进行测试，在测试 <code>GenerateInvoiceEmail()</code> 函数之前，我们必须先设置好调用 <code>GetFeatureToggleRouter()</code> 与 <code>NewFeatureDecision()</code> 函数的环境，才能确保可以到达待测试逻辑代码块。
2. 随着项目功能模块的增多，每个模块都与 <code>FeatureDecisions</code> 模块发生了耦合，使得该模块变成了全局依赖模块。

```go
// invoice_emailler.go

type EmaillerDecisions interface {
    IncludeOrderCancellationInEmail() bool
}

type InvoiceEmailler struct {
    invoice Invoice
    decisions EmaillerDecisions
}

func (emailler *InvoiceEmailler) SetDecisions(decisions *FeatureDecisions) {
    emailler.decisions = decisions
}

func (emailler *InvoiceEmailler) GenerateInvoiceEmail() Email {
    baseEmail := buildEmailForInvoice(emailler.invoice)
    if emailler.decisions.includeOrderCancellationInEmail() {
        return addOrderCallationContentToEmail(baseEmail)
    } else {
        return baseEmail
    }
}

// inject decisions module
func NewInvoiceEmailler(decisions EmaillerDecisions， invoice Invoice) *InvoiceEmailler {
    return &InvoiceEmailler{
        invoice: invoice,
        decisions: decisions,
    }
}

```

此处“坏味道”的根本原因是业务模块对 <code>FeatureDecisions</code> 模块的依赖，通过**控制反转**在 <code>InvoiceEmailler</code> 模块创建时，将 <code>FeatureDecisions</code> 注入到 <code>InvoiceEmailler</code> 中，就可以消除业务模块对与 <code>FeatureDecisions</code> 模块的依赖。

### 消除条件判断

到现在位置我们代码已经获得了很大优化，那我们能不能在进一步消除 <code>GenerateInvoiceEmail()</code> 函数的条件判断呢？ 对于简单的场景下的条件判断语句是没有什么问题的，但是在复杂的业务逻辑中，太多的条件判断并不利于代码的维护。比如如果配置中包含 feature "after-sales-service",  需要在 email 中添加售后服务的信息。现在我们必须回到  <code>GenerateInvoiceEmail()</code>，修改判断条件再加上添加售后服务的信息。

通过策略模式，我们可以进一步的消除业务代码中的逻辑判断，提升代码的可扩展性。

```go

type EmaillerDecisions interface {
	IncludeOrderCancellationInEmail() bool

	IncludeAfterSalesServiceInEmail() bool
}

type AdditionalContentEnhancer interface {
	EnhanceContent(Email) Email
}

type InvoiceEmailler struct {
	invoice  Invoice
	enhancer AdditionalContentEnhancer
}

func (emailler *InvoiceEmailler) GenerateInvoiceEmail() Email {
	baseEmail := buildEmailForInvoice(emailler.invoice)
	return enhancer.EnhanceContent(baseEmail)
}

type OrderCallationContentEnhancer struct{}

func (e OrderCallationContentEnhancer) addOrderCallationContentToEmail(email Email) (newEmail Email) {
	// implement addOrderCallationContentToEmail
	return
}

func (e OrderCallationContentEnhancer) EnhanceContent(email Email) Email {
	return e.addOrderCallationContentToEmail(email)
}

type AfterSalesServiceContentEnhancer struct{}

func (e AfterSalesServiceContentEnhancer) addAfterSalesServiceContentToEmail(email Email) (newEmail Email) {
	// implement addAfterSalesServiceContentToEmail
	return
}

func (e AfterSalesServiceContentEnhancer) EnhanceContent(email Email) Email {
	return e.addAfterSalesServiceContentToEmail(email)
}

type OrderCallationAndAfterSalesServiceContentEnhancer struct {
	OrderCallationContentEnhancer
	AfterSalesServiceContentEnhancer
}

func (e OrderCallationAndAfterSalesServiceContentEnhancer) EnhanceContent(email Email) Email {
	email = e.OrderCallationContentEnhancer.EnhanceContent(email)
	email = e.AfterSalesServiceContentEnhancer.EnhanceContent(email)
	return email
}

type IdentityContentEnhancer struct{}

func (e IdentityContentEnhancer) EnhanceContent(email Email) Email {
	return email
}

// inject decisions module
func NewInvoiceEmailler(decisions EmaillerDecisions, invoice Invoice) *InvoiceEmailler {
	emailler := &InvoiceEmailler{
		invoice: invoice,
	}

	if decisions.IncludeOrderCancellationInEmail() && decisions.IncludeAfterSalesServiceInEmail() {
		emailler.enhancer = OrderCallationAndAfterSalesServiceContentEnhancer{}
	} else if decisions.IncludeOrderCancellationInEmail() {
		emailler.enhancer = OrderCallationContentEnhancer{}
	} else if decisions.IncludeAfterSalesServiceInEmail() {
		emailler.enhancer = AfterSalesServiceContentEnhancer{}
	} else {
		emailler.enhancer = IdentityContentEnhancer{}
	}

	return
}

```

到现在我们就差不多完成了对于代码的优化。
